"""Evaluate FluxFlow reconstructions and write one JSON metrics report.

PSNR and SSIM are computed on clipped normalized images. Source-flux error
uses clipped fluxes, whereas source detection uses unclipped, finite fluxes.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import sep
import torch
import torch.nn.functional as F
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader

from data.dataset import FMDataset, denormalize_hst, load_sources
from models.unet import FMUNet
from models.unet_v2 import UNetV2

from infer import (
    make_gaussian_kernel,
    sample_euler_mcfm,
    sample_midpoint_mcfm,
)
from inverse_solver_samplers import SAMPLERS


def _gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k1 = g / g.sum()
    return np.outer(k1, k1)


def _conv2d(img, kernel):
    kh, kw = kernel.shape
    windows = sliding_window_view(img, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def psnr_norm(pred, gt, max_val=1.0):
    mse = np.mean((pred - gt) ** 2)
    return 10.0 * np.log10((max_val ** 2) / (mse + 1e-12))


def ssim_norm(pred, gt, max_val=1.0, win_size=11, sigma=1.5,
              k1=0.01, k2=0.03):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    kernel = _gaussian_kernel(win_size, sigma)
    mu_p = _conv2d(pred, kernel)
    mu_g = _conv2d(gt, kernel)
    mu_p2 = mu_p * mu_p
    mu_g2 = mu_g * mu_g
    mu_pg = mu_p * mu_g
    sigma_p2 = _conv2d(pred * pred, kernel) - mu_p2
    sigma_g2 = _conv2d(gt * gt, kernel) - mu_g2
    sigma_pg = _conv2d(pred * gt, kernel) - mu_pg
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    num = (2 * mu_pg + c1) * (2 * sigma_pg + c2)
    den = (mu_p2 + mu_g2 + c1) * (sigma_p2 + sigma_g2 + c2)
    return float(np.mean(num / den))


def src_flux_l1(pred_clipped, gt_clipped, sources):
    if sources is None or len(sources["x"]) == 0:
        return 0.0
    gt_c = np.ascontiguousarray(gt_clipped, dtype=np.float64)
    pred_c = np.ascontiguousarray(pred_clipped, dtype=np.float64)
    gt_sum, _, _ = sep.sum_ellipse(gt_c, sources["x"], sources["y"],
                                   sources["a"], sources["b"], sources["theta"], 2.5)
    pred_sum, _, _ = sep.sum_ellipse(pred_c, sources["x"], sources["y"],
                                     sources["a"], sources["b"], sources["theta"], 2.5)
    return float(np.abs(pred_sum - gt_sum).sum())


def filter_mask_by_area(gt_mask, min_area):
    if min_area <= 1:
        return gt_mask
    labels, counts = np.unique(gt_mask, return_counts=True)
    keep = {int(l) for l, c in zip(labels, counts) if l > 0 and c >= min_area}
    return np.where(np.isin(gt_mask, list(keep)), gt_mask, 0).astype(gt_mask.dtype)


def gt_centroids(gt_mask):
    labels = np.unique(gt_mask); labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros(0), np.zeros(0)
    cx, cy = [], []
    for lab in labels:
        ys, xs = np.where(gt_mask == lab)
        cx.append(xs.mean()); cy.append(ys.mean())
    return np.asarray(cx), np.asarray(cy)


def detect_xy(img, thresh, minarea, deblend_nthresh=32, deblend_cont=0.02):
    img = np.ascontiguousarray(img.astype(np.float64))
    bkg = sep.Background(img)
    try:
        obj = sep.extract(img - bkg, thresh=thresh, err=bkg.globalrms,
                          minarea=minarea, deblend_nthresh=deblend_nthresh,
                          deblend_cont=deblend_cont)
    except Exception:
        return np.empty(0), np.empty(0)
    return np.asarray(obj["x"]), np.asarray(obj["y"])


def greedy_match(px, py, gx, gy, tol):
    n_pred, n_gt = len(px), len(gx)
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    px = np.asarray(px, float); py = np.asarray(py, float)
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    d2 = (px[:, None] - gx[None, :]) ** 2 + (py[:, None] - gy[None, :]) ** 2
    within = d2 <= tol * tol
    if not within.any():
        return 0, n_pred, n_gt
    idxs = np.argwhere(within)
    order = np.argsort(d2[within])
    used_p = np.zeros(n_pred, dtype=bool)
    used_g = np.zeros(n_gt, dtype=bool)
    tp = 0
    for k in order:
        i, j = int(idxs[k, 0]), int(idxs[k, 1])
        if used_p[i] or used_g[j]:
            continue
        used_p[i] = True; used_g[j] = True
        tp += 1
    return tp, n_pred - tp, n_gt - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def clean(img):
    finite = img[np.isfinite(img)]
    vmax = finite.max() if finite.size > 0 else 0.0
    return np.nan_to_num(img, nan=0.0, posinf=vmax, neginf=0.0)


def parse_args():
    p = argparse.ArgumentParser(
        description="FluxFlow inference, reconstruction evaluation, and source detection")
    p.add_argument("--config", required=True)
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--out_json", required=True,
                   help="Path to write the per-config JSON result.")
    p.add_argument("--tag", required=True,
                   help="Identifier (key) for this config in the JSON.")

    # Sampling parameters.
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    p.add_argument("--inverse_solver",
                   choices=["none", "dps_style", "pigdm_style",
                            "flowdps", "flower"],
                   default="none")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker count; use 0 on network filesystems.")
    p.add_argument("-n", "--num_samples", type=int, default=None)
    p.add_argument("--bridge_sigma", type=float, default=None)

    # Measurement-consistency parameters.
    p.add_argument("--correction_mode",
                   choices=["wiener", "adjoint", "none"], default="wiener")
    p.add_argument("--adjoint_mode",
                   choices=["nearest", "bicubic"], default="nearest")
    p.add_argument("--no_adjoint_scale", action="store_true")
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--eta_schedule",
                   choices=["constant", "linear_decay", "cosine"],
                   default="linear_decay")
    p.add_argument("--psf_sigma", type=float, default=2.0)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--snr", type=float, default=50.0)
    p.add_argument("--max_gain", type=float, default=0.5)
    p.add_argument("--no_mcfm", action="store_true")
    p.add_argument("--guide_scale", type=float, default=0.5)
    p.add_argument("--guide_schedule",
                   choices=["constant", "linear_decay", "cosine"],
                   default="linear_decay")
    p.add_argument("--pinv_reg", type=float, default=1e-2)
    p.add_argument("--cg_iters", type=int, default=12)
    p.add_argument("--stochasticity", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--flower_gamma", type=float, default=0.0)

    # Source-detection parameters.
    p.add_argument("--det_thresh", type=float, default=4.0)
    p.add_argument("--det_minarea", type=int, default=5)
    p.add_argument("--det_deblend_nthresh", type=int, default=32)
    p.add_argument("--det_deblend_cont", type=float, default=0.02)
    p.add_argument("--det_gt_min_area", type=int, default=10)
    p.add_argument("--det_match_tol", type=float, default=16.0)

    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg.get("train", {})

    bridge_sigma = (args.bridge_sigma if args.bridge_sigma is not None
                    else float(tcfg.get("bridge_sigma", 1.0)))
    scale = args.scale if args.scale is not None else int(
        dcfg.get("scale_factor", 4))
    apply_scale = not args.no_adjoint_scale

    arch = mcfg.get("arch", "fmunet")
    if arch == "unet_v2":
        model = UNetV2(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            base_channels=mcfg.get("base_channels", 64),
            channel_mults=tuple(mcfg.get("channel_mults", [1, 2, 4])),
            num_res_blocks=mcfg.get("num_res_blocks", 2),
            time_dim=mcfg["time_dim"],
        ).to(device)
    else:
        model = FMUNet(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            base_dim=mcfg["base_dim"],
            time_dim=mcfg["time_dim"],
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights (no EMA found)")
    model.eval()

    use_correction = ((not args.no_mcfm)
                      and (args.correction_mode != "none"
                           or args.inverse_solver != "none"))
    psf_kernel = None
    if use_correction:
        psf_kernel = make_gaussian_kernel(args.psf_sigma, channels=1).to(device)

    test_ds = FMDataset(dcfg["data_dir"], split="test",
                        split_file=dcfg["split_file"])
    if args.num_samples is not None:
        test_ds.sample_names = test_ds.sample_names[:args.num_samples]
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Load normalization statistics for flux-domain metrics.
    with open(Path(dcfg["data_dir"]) / "normalize.json") as f:
        nstats = json.load(f)
    vmin = nstats["hst_sci"]["min"]
    vmax = nstats["hst_sci"]["max"]
    vrng = vmax - vmin

    print(f"FluxFlow evaluation [{args.tag}]")
    print(f"  Samples: {len(test_ds)}, Steps: {args.num_steps}, Solver: {args.solver}")
    print(f"  scale={scale}  bridge_sigma={bridge_sigma}")
    print(f"  eta={args.eta} ({args.eta_schedule}) psf_sigma={args.psf_sigma} "
          f"snr={args.snr} max_gain={args.max_gain} "
          f"correction={args.correction_mode} adjoint={args.adjoint_mode}"
          f"{'+1/s^2' if apply_scale else ''}")
    if args.inverse_solver != "none":
        print(f"  inverse_solver={args.inverse_solver} "
              f"guide_scale={args.guide_scale} pinv_reg={args.pinv_reg} "
              f"cg_iters={args.cg_iters} noise_std={args.noise_std}")

    torch.manual_seed(args.seed)
    sample_fn = (SAMPLERS[args.inverse_solver]
                 if args.inverse_solver != "none"
                 else (sample_euler_mcfm if args.solver == "euler"
                       else sample_midpoint_mcfm))

    psnrs, ssims, src_l1s = [], [], []
    tot_tp = tot_fp = tot_fn = 0
    per_p, per_r, per_f1 = [], [], []
    n_pred_total = n_gt_total = 0
    n_skipped = 0
    sample_idx = 0

    data_dir = Path(dcfg["data_dir"])

    for desi, hst_gt, _wht, _seg in test_loader:
        desi = desi.to(device)
        hst_gt = hst_gt.to(device)
        B = desi.shape[0]
        hst_size = (hst_gt.shape[-2], hst_gt.shape[-1])

        if args.inverse_solver != "none":
            pred_norm = sample_fn(
                model, desi, args.num_steps, device,
                hst_size=hst_size, psf_kernel=psf_kernel, scale=scale,
                guide_scale=args.guide_scale,
                guide_schedule=args.guide_schedule,
                pinv_reg=args.pinv_reg, cg_iters=args.cg_iters,
                stochasticity=args.stochasticity,
                noise_std=args.noise_std,
                flower_gamma=args.flower_gamma,
            )
        else:
            pred_norm = sample_fn(
                model, desi, args.num_steps, device,
                hst_size=hst_size, bridge_sigma=bridge_sigma,
                psf_kernel=psf_kernel, scale=scale,
                eta=args.eta, eta_schedule=args.eta_schedule,
                snr=args.snr, max_gain=args.max_gain,
                correction_mode=args.correction_mode,
                adjoint_mode=args.adjoint_mode,
                apply_scale=apply_scale,
                no_mcfm=args.no_mcfm,
                desi_mc=None,
            )

        pred_np = pred_norm.cpu().numpy()  # (B,1,H,W) in normalized domain
        gt_np = hst_gt.cpu().numpy()

        for j in range(B):
            name = test_ds.sample_names[sample_idx]

            pred_n = pred_np[j, 0]
            gt_n = gt_np[j, 0]

            pred_n_clip = np.clip(clean(pred_n), 0.0, 1.0)
            gt_n_clip = np.clip(clean(gt_n), 0.0, 1.0)

            psnrs.append(psnr_norm(pred_n_clip, gt_n_clip, max_val=1.0))
            ssims.append(ssim_norm(pred_n_clip, gt_n_clip, max_val=1.0))

            pred_flux_clip = pred_n_clip * vrng + vmin
            gt_flux_clip = gt_n_clip * vrng + vmin

            sources = load_sources(dcfg["data_dir"], name)
            src_l1s.append(src_flux_l1(pred_flux_clip, gt_flux_clip, sources))

            # Perform detection on finite but otherwise unclipped fluxes.
            pred_flux = clean(pred_n) * vrng + vmin
            mask_path = data_dir / name / "hst_masks.npy"
            if mask_path.exists():
                gt_mask = filter_mask_by_area(np.load(mask_path),
                                              args.det_gt_min_area)
                gx, gy = gt_centroids(gt_mask)
                px, py = detect_xy(pred_flux, args.det_thresh,
                                   args.det_minarea,
                                   args.det_deblend_nthresh,
                                   args.det_deblend_cont)
                tp, fp, fn = greedy_match(px, py, gx, gy,
                                          tol=args.det_match_tol)
                tot_tp += tp; tot_fp += fp; tot_fn += fn
                n_pred_total += len(px)
                n_gt_total += tp + fn
                p_, r_, f1_ = prf(tp, fp, fn)
                per_p.append(p_); per_r.append(r_); per_f1.append(f1_)
            else:
                n_skipped += 1

            if (sample_idx + 1) % args.log_every == 0 or sample_idx == 0:
                base_line = (f"  [{sample_idx+1}/{len(test_ds)}] {name}  "
                             f"PSNR={psnrs[-1]:.2f}  SSIM={ssims[-1]:.4f}  "
                             f"SrcL1={src_l1s[-1]:.4f}")
                if per_f1:
                    base_line += (f"  P/R/F1={per_p[-1]:.3f}/"
                                  f"{per_r[-1]:.3f}/{per_f1[-1]:.3f}")
                print(base_line, flush=True)

            sample_idx += 1

    P_micro, R_micro, F1_micro = prf(tot_tp, tot_fp, tot_fn)

    result = {
        "tag": args.tag,
        "scale": scale,
        "snr": args.snr,
        "psf_sigma": args.psf_sigma,
        "eta": args.eta,
        "eta_schedule": args.eta_schedule,
        "max_gain": args.max_gain,
        "num_steps": args.num_steps,
        "solver": args.solver,
        "inverse_solver": args.inverse_solver,
        "seed": args.seed,
        "correction_mode": args.correction_mode,
        "adjoint_mode": args.adjoint_mode,
        "apply_scale": apply_scale,
        "n_samples_eval": len(psnrs),
        "n_samples_det": len(per_f1),
        "n_skipped_det": n_skipped,
        "mean_psnr": float(np.mean(psnrs)) if psnrs else None,
        "mean_ssim": float(np.mean(ssims)) if ssims else None,
        "mean_src_flux_l1": float(np.mean(src_l1s)) if src_l1s else None,
        "det_micro_precision": P_micro,
        "det_micro_recall": R_micro,
        "det_micro_f1": F1_micro,
        "det_macro_precision": float(np.mean(per_p)) if per_p else None,
        "det_macro_recall": float(np.mean(per_r)) if per_r else None,
        "det_macro_f1": float(np.mean(per_f1)) if per_f1 else None,
        "det_total_tp": int(tot_tp),
        "det_total_fp": int(tot_fp),
        "det_total_fn": int(tot_fn),
        "det_n_pred_total": int(n_pred_total),
        "det_n_gt_total": int(n_gt_total),
        "det_thresh": args.det_thresh,
        "det_minarea": args.det_minarea,
        "det_match_tol": args.det_match_tol,
        "det_gt_min_area": args.det_gt_min_area,
        "guide_scale": args.guide_scale,
        "guide_schedule": args.guide_schedule,
        "pinv_reg": args.pinv_reg,
        "cg_iters": args.cg_iters,
        "stochasticity": args.stochasticity,
        "noise_std": args.noise_std,
        "flower_gamma": args.flower_gamma,
    }

    print(f"\n--- Results [{args.tag}] ---")
    print(f"  PSNR={result['mean_psnr']:.4f}  SSIM={result['mean_ssim']:.4f}  "
          f"SrcL1={result['mean_src_flux_l1']:.6f}")
    print(f"  Det micro P={P_micro:.4f} R={R_micro:.4f} F1={F1_micro:.4f}")
    print(f"  Det macro P={result['det_macro_precision']:.4f} "
          f"R={result['det_macro_recall']:.4f} F1={result['det_macro_f1']:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
