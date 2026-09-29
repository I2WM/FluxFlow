"""Measurement-consistent flow-matching inference for DESI-to-HST super-resolution.

The forward model is Gaussian PSF blur followed by area downsampling.  The
default correction lifts the low-resolution residual with the exact adjoint
of area downsampling and applies Wiener regularization on the high-resolution
grid.  The ``adjoint`` mode provides a direct ``A^T r`` baseline.
"""

import argparse
import math
import os

import numpy as np
import sep
import torch
import torch.nn.functional as F
import yaml
from astropy.io import fits
from torch.utils.data import DataLoader

from data.dataset import FMDataset, denormalize_hst, load_sources
from models.unet import FMUNet
from models.unet_v2 import UNetV2


def make_gaussian_kernel(sigma: float, channels: int = 1) -> torch.Tensor:
    """Create a 2D Gaussian blur kernel (sum-normalized)."""
    radius = int(math.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32) - radius
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.expand(channels, 1, size, size).clone()


def downsample_blur(x_hst: torch.Tensor, psf_kernel: torch.Tensor,
                    scale: int = 4) -> torch.Tensor:
    """Forward model A: HR-scale Gaussian PSF blur + area downsample."""
    C = x_hst.shape[1]
    pad = psf_kernel.shape[-1] // 2
    x_blurred = F.conv2d(x_hst, psf_kernel, padding=pad, groups=C)
    x_low = F.interpolate(x_blurred, scale_factor=1.0 / scale, mode="area")
    return x_low


def apply_adjoint_upsample(residual_lr: torch.Tensor, scale: int, *,
                            mode: str = "nearest",
                            hr_size=None,
                            apply_scale: bool = True) -> torch.Tensor:
    """Apply D_s^T to lift an LR residual back to the HR grid.

    For area-pool downsampling with factor s, the strict adjoint is
        D_s^T(y)[m,n] = (1/s^2) * y[floor(m/s), floor(n/s)]
    i.e. nearest-neighbor replication with a 1/s^2 magnitude factor.

    Args:
        residual_lr: (B, C, H_lr, W_lr) LR-space residual.
        scale: downsampling factor s.
        mode: "nearest" applies the area-pooling adjoint. "bicubic" applies
              an interpolation-based approximation.
        hr_size: (H, W) of HR target. If None, use scale_factor=scale.
        apply_scale: if True (default), multiply by 1/s^2 (mass-preserving
              adjoint). Set False to reproduce v2's bicubic-without-scaling.
    """
    if mode == "nearest":
        if hr_size is not None:
            up = F.interpolate(residual_lr, size=hr_size, mode="nearest")
        else:
            up = F.interpolate(residual_lr, scale_factor=float(scale),
                               mode="nearest")
    elif mode == "bicubic":
        if hr_size is not None:
            up = F.interpolate(residual_lr, size=hr_size,
                               mode="bicubic", align_corners=False)
        else:
            up = F.interpolate(residual_lr, scale_factor=float(scale),
                               mode="bicubic", align_corners=False)
    else:
        raise ValueError(f"Unknown adjoint mode: {mode}")

    if apply_scale:
        up = up / (scale ** 2)
    return up


def apply_AT(residual_lr: torch.Tensor, psf_kernel: torch.Tensor,
             scale: int, *, adjoint_mode: str = "nearest",
             hr_size=None, apply_scale: bool = True) -> torch.Tensor:
    """Apply A^T r_lr = H^T D_s^T r_lr.

    The Gaussian PSF used here is symmetric, so H^T == H and we just
    convolve the upsampled residual with the same kernel.
    """
    r_hr = apply_adjoint_upsample(residual_lr, scale, mode=adjoint_mode,
                                   hr_size=hr_size, apply_scale=apply_scale)
    C = r_hr.shape[1]
    pad = psf_kernel.shape[-1] // 2
    return F.conv2d(r_hr, psf_kernel, padding=pad, groups=C)


def wiener_deconv_hr(y_hr: torch.Tensor, psf_kernel: torch.Tensor,
                      snr: float = 100.0,
                      max_gain: float = 0.0) -> torch.Tensor:
    """Wiener-regularized deconvolution in HR frequency domain.

    Built and applied at the same (HR) resolution: zero-pad the HR-scale
    PSF to (H_hr, W_hr), build W(k) = H*(k) / (|H(k)|^2 + 1/SNR), then
    multiply with FFT(y_hr).
    """
    B, C, H, W = y_hr.shape
    psf_2d = psf_kernel[0, 0]
    psf_padded = torch.zeros(H, W, device=y_hr.device, dtype=y_hr.dtype)
    kH, kW = psf_2d.shape
    psf_padded[:kH, :kW] = psf_2d
    psf_padded = torch.roll(psf_padded, shifts=(-kH // 2, -kW // 2),
                             dims=(0, 1))

    H_fft = torch.fft.rfft2(psf_padded)
    H_conj = H_fft.conj()
    H_abs2 = H_fft.abs() ** 2
    wiener_filter = H_conj / (H_abs2 + 1.0 / snr)

    if max_gain > 0:
        gain = wiener_filter.abs()
        phase = torch.angle(wiener_filter)
        wiener_filter = torch.clamp(gain, max=max_gain) * torch.exp(1j * phase)

    out = torch.zeros_like(y_hr)
    for c in range(C):
        Y_fft = torch.fft.rfft2(y_hr[:, c])
        out[:, c] = torch.fft.irfft2(Y_fft * wiener_filter, s=(H, W))
    return out


def compute_correction_hr(x_candidate: torch.Tensor,
                           desi_mc: torch.Tensor,
                           psf_kernel: torch.Tensor,
                           scale: int,
                           hr_size,
                           *,
                           correction_mode: str = "wiener",
                           adjoint_mode: str = "nearest",
                           apply_scale: bool = True,
                           snr: float = 100.0,
                           max_gain: float = 0.0) -> torch.Tensor:
    """HR-space MC-FM correction.

    correction_mode:
      - "wiener": D_s^T -> Wiener_HR (paper's MC-FS, fixed to act in HR).
      - "adjoint": A^T r_lr (DPS / Pi-GDM-style ablation), no deconvolution.
    """
    x_projected = downsample_blur(x_candidate, psf_kernel, scale)
    residual_lr = x_projected - desi_mc

    if correction_mode == "adjoint":
        return apply_AT(residual_lr, psf_kernel, scale,
                        adjoint_mode=adjoint_mode,
                        hr_size=hr_size, apply_scale=apply_scale)

    if correction_mode != "wiener":
        raise ValueError(f"Unknown correction_mode: {correction_mode}")

    residual_hr = apply_adjoint_upsample(residual_lr, scale,
                                          mode=adjoint_mode,
                                          hr_size=hr_size,
                                          apply_scale=apply_scale)
    return wiener_deconv_hr(residual_hr, psf_kernel, snr=snr,
                             max_gain=max_gain)


def nlm_denoise(image_np: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means denoising to a 2D image."""
    from skimage.restoration import denoise_nl_means, estimate_sigma

    sigma_est = np.mean(estimate_sigma(image_np))
    denoised = denoise_nl_means(
        image_np, h=1.15 * sigma_est, fast_mode=True,
        patch_size=5, patch_distance=6,
    )
    return denoised


def sample_euler_mcfm(model, desi, num_steps, device, *,
                       hst_size=None,
                       bridge_sigma=1.0,
                       psf_kernel=None, scale=4,
                       eta=1.0, eta_schedule="constant",
                       snr=100.0, max_gain=0.0,
                       correction_mode="wiener",
                       adjoint_mode="nearest",
                       apply_scale=True,
                       no_mcfm=False,
                       desi_mc=None):
    """Integrate the conditional velocity field with Euler updates."""
    if desi_mc is None:
        desi_mc = desi

    B = desi.shape[0]
    H, W = hst_size if hst_size else (512, 512)

    desi_up = F.interpolate(desi, size=(H, W), mode="bicubic",
                             align_corners=False)
    noise = torch.randn(B, 1, H, W, device=device)
    x = (1 - bridge_sigma) * desi_up + bridge_sigma * noise

    dt = 1.0 / num_steps
    skip_corr = no_mcfm or psf_kernel is None or correction_mode == "none"

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)

        with torch.no_grad():
            v = model(x, t, desi)

        if skip_corr:
            x = x + v * dt
        else:
            with torch.no_grad():
                x_candidate = x + v * dt
                correction_hr = compute_correction_hr(
                    x_candidate, desi_mc, psf_kernel, scale,
                    hr_size=(H, W),
                    correction_mode=correction_mode,
                    adjoint_mode=adjoint_mode,
                    apply_scale=apply_scale,
                    snr=snr, max_gain=max_gain,
                )
                eta_t = _get_eta(eta, i, num_steps, eta_schedule)
                x = x_candidate - eta_t * correction_hr

    return x


def sample_midpoint_mcfm(model, desi, num_steps, device, *,
                          hst_size=None,
                          bridge_sigma=1.0,
                          psf_kernel=None, scale=4,
                          eta=1.0, eta_schedule="constant",
                          snr=100.0, max_gain=0.0,
                          correction_mode="wiener",
                          adjoint_mode="nearest",
                          apply_scale=True,
                          no_mcfm=False,
                          desi_mc=None):
    """Integrate the conditional velocity field with midpoint updates."""
    if desi_mc is None:
        desi_mc = desi

    B = desi.shape[0]
    H, W = hst_size if hst_size else (512, 512)

    desi_up = F.interpolate(desi, size=(H, W), mode="bicubic",
                             align_corners=False)
    noise = torch.randn(B, 1, H, W, device=device)
    x = (1 - bridge_sigma) * desi_up + bridge_sigma * noise

    dt = 1.0 / num_steps
    skip_corr = no_mcfm or psf_kernel is None or correction_mode == "none"

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        t_mid = torch.full((B,), (i + 0.5) * dt, device=device)

        with torch.no_grad():
            v1 = model(x, t, desi)
            x_mid = x + v1 * (dt / 2)
            v2 = model(x_mid, t_mid, desi)

        if skip_corr:
            x = x + v2 * dt
        else:
            with torch.no_grad():
                x_candidate = x + v2 * dt
                correction_hr = compute_correction_hr(
                    x_candidate, desi_mc, psf_kernel, scale,
                    hr_size=(H, W),
                    correction_mode=correction_mode,
                    adjoint_mode=adjoint_mode,
                    apply_scale=apply_scale,
                    snr=snr, max_gain=max_gain,
                )
                eta_t = _get_eta(eta, i, num_steps, eta_schedule)
                x = x_candidate - eta_t * correction_hr

    return x


def _get_eta(eta_base: float, step: int, total_steps: int,
             schedule: str) -> float:
    if schedule == "constant":
        return eta_base
    frac = step / max(total_steps - 1, 1)
    if schedule == "linear_decay":
        return eta_base * (1.0 - frac)
    if schedule == "cosine":
        return eta_base * 0.5 * (1.0 + math.cos(math.pi * frac))
    return eta_base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measurement-consistent flow-matching inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--solver", choices=["euler", "midpoint"],
                        default="euler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count; use 0 on network filesystems.")
    parser.add_argument("-n", "--num_samples", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--denoise", action="store_true",
                        help="Apply NLM denoising to DESI input before using "
                             "as guidance and MC-FM target.")

    parser.add_argument("--bridge_sigma", type=float, default=None)

    mcfm = parser.add_argument_group("measurement-consistency parameters")
    mcfm.add_argument("--correction_mode",
                      choices=["wiener", "adjoint", "none"],
                      default="wiener",
                      help="wiener=regularized inverse; adjoint=A^T r; none=no correction.")
    mcfm.add_argument("--adjoint_mode",
                      choices=["nearest", "bicubic"],
                      default="nearest",
                      help="Residual lifting method; nearest is the area-pool adjoint.")
    mcfm.add_argument("--no_adjoint_scale", action="store_true",
                      help="Do not apply the 1/s^2 factor in the downsampling adjoint.")
    mcfm.add_argument("--eta", type=float, default=1.0,
                      help="Measurement-consistency correction step size.")
    mcfm.add_argument("--eta_schedule",
                      choices=["constant", "linear_decay", "cosine"],
                      default="linear_decay")
    mcfm.add_argument("--psf_sigma", type=float, default=2.0,
                      help="Gaussian PSF sigma (HR pixels).")
    mcfm.add_argument("--scale", type=int, default=None)
    mcfm.add_argument("--snr", type=float, default=100.0)
    mcfm.add_argument("--max_gain", type=float, default=0.0)
    mcfm.add_argument("--no_mcfm", action="store_true",
                      help="Disable the measurement-consistency correction.")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
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

    use_correction = (not args.no_mcfm) and args.correction_mode != "none"
    psf_kernel = None
    if use_correction:
        psf_kernel = make_gaussian_kernel(args.psf_sigma,
                                           channels=1).to(device)

    test_ds = FMDataset(dcfg["data_dir"], split="test",
                        split_file=dcfg["split_file"])
    if args.num_samples is not None:
        test_ds.sample_names = test_ds.sample_names[:args.num_samples]
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if not use_correction:
        mcfm_mode = "disabled"
    else:
        mcfm_mode = (
            f"mode={args.correction_mode}, adjoint={args.adjoint_mode}"
            f"{'+1/s^2' if apply_scale else ''}, "
            f"eta={args.eta} ({args.eta_schedule}), "
            f"psf_sigma={args.psf_sigma}, "
            f"snr={args.snr}, max_gain={args.max_gain}"
        )
    print("Measurement-consistent flow-matching inference")
    print(f"  Samples: {len(test_ds)}, Steps: {args.num_steps}, "
          f"Solver: {args.solver}")
    print(f"  Bridge sigma: {bridge_sigma}, Scale: {scale}")
    print(f"  MC-FM: {mcfm_mode}")
    print(f"  Denoise: {args.denoise}")

    torch.manual_seed(args.seed)

    sample_fn = (sample_euler_mcfm if args.solver == "euler"
                 else sample_midpoint_mcfm)

    if args.save_dir is not None:
        args.save = True
    save_dir = None
    if args.save:
        if args.save_dir is not None:
            save_dir = args.save_dir
        else:
            suffix = "_denoised" if args.denoise else ""
            tag = f"{args.correction_mode}_{args.adjoint_mode}"
            save_dir = os.path.join(
                os.path.dirname(args.checkpoint), "..",
                f"predictions_mcfm_{tag}{suffix}",
            )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving to: {save_dir}")

    all_psnr, all_mse = [], []
    all_flux_err, all_src_flux_err = [], []
    sample_idx = 0

    for desi, hst_gt, _wht, _seg in test_loader:
        desi = desi.to(device)
        hst_gt = hst_gt.to(device)
        B = desi.shape[0]

        hst_size = (hst_gt.shape[-2], hst_gt.shape[-1])

        desi_mc = None
        if args.denoise:
            desi_np = desi.cpu().numpy()
            desi_denoised_np = np.empty_like(desi_np)
            for j in range(B):
                desi_denoised_np[j, 0] = nlm_denoise(desi_np[j, 0])
            desi_mc = torch.from_numpy(desi_denoised_np).to(device)

        pred_norm = sample_fn(
            model, desi, args.num_steps, device,
            hst_size=hst_size,
            bridge_sigma=bridge_sigma,
            psf_kernel=psf_kernel,
            scale=scale,
            eta=args.eta,
            eta_schedule=args.eta_schedule,
            snr=args.snr,
            max_gain=args.max_gain,
            correction_mode=args.correction_mode,
            adjoint_mode=args.adjoint_mode,
            apply_scale=apply_scale,
            no_mcfm=args.no_mcfm,
            desi_mc=desi_mc,
        )

        pred_np = pred_norm.cpu().numpy()
        gt_np = hst_gt.cpu().numpy()

        pred_flux = denormalize_hst(pred_np, dcfg["data_dir"])
        gt_flux = denormalize_hst(gt_np, dcfg["data_dir"])

        for j in range(B):
            name = test_ds.sample_names[sample_idx]

            if args.eval:
                mse = np.mean((pred_np[j] - gt_np[j]) ** 2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-12))
                all_psnr.append(psnr)
                all_mse.append(mse)

                pred_img = pred_flux[j, 0]
                gt_img = gt_flux[j, 0]
                flux_err = abs(pred_img.sum() - gt_img.sum())
                all_flux_err.append(flux_err)

                sources = load_sources(dcfg["data_dir"], name)
                src_flux_err = 0.0
                if sources is not None and len(sources["x"]) > 0:
                    gt_c = np.ascontiguousarray(gt_img, dtype=np.float64)
                    pred_c = np.ascontiguousarray(pred_img, dtype=np.float64)
                    x, y = sources["x"], sources["y"]
                    a, b = sources["a"], sources["b"]
                    theta = sources["theta"]
                    r = 2.5
                    gt_sum, _, _ = sep.sum_ellipse(gt_c, x, y, a, b, theta, r)
                    pred_sum, _, _ = sep.sum_ellipse(pred_c, x, y, a, b,
                                                     theta, r)
                    src_flux_err = float(np.abs(pred_sum.sum() - gt_sum.sum()))
                all_src_flux_err.append(src_flux_err)

                print(
                    f"  [{sample_idx + 1}/{len(test_ds)}] {name}  "
                    f"PSNR: {psnr:.2f}dB  MSE: {mse:.2e}  "
                    f"FluxErr: {flux_err:.4f}  SrcFluxErr: {src_flux_err:.4f}"
                )

            if args.save and save_dir is not None:
                fits.writeto(
                    os.path.join(save_dir, f"{name}_pred.fits"),
                    pred_flux[j, 0].astype(np.float32), overwrite=True,
                )
                fits.writeto(
                    os.path.join(save_dir, f"{name}_gt.fits"),
                    gt_flux[j, 0].astype(np.float32), overwrite=True,
                )
                if args.denoise and desi_mc is not None:
                    desi_orig_flux = denormalize_hst(
                        desi.cpu().numpy(), dcfg["data_dir"]
                    )
                    desi_den_flux = denormalize_hst(
                        desi_mc.cpu().numpy(), dcfg["data_dir"]
                    )
                    fits.writeto(
                        os.path.join(save_dir, f"{name}_desi_orig.fits"),
                        desi_orig_flux[j, 0].astype(np.float32),
                        overwrite=True,
                    )
                    fits.writeto(
                        os.path.join(save_dir, f"{name}_desi_denoised.fits"),
                        desi_den_flux[j, 0].astype(np.float32),
                        overwrite=True,
                    )

            sample_idx += 1

    if args.eval and all_psnr:
        print(f"\n--- Results ({len(all_psnr)} samples) ---")
        print(f"  Mean PSNR:        {np.mean(all_psnr):.2f} dB")
        print(f"  Mean MSE:         {np.mean(all_mse):.2e}")
        print(f"  Mean Flux Error:  {np.mean(all_flux_err):.4f}")
        print(f"  Mean Src Flux Err:{np.mean(all_src_flux_err):.4f}")


if __name__ == "__main__":
    main()
