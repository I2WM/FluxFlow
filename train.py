"""OT-CFM training for DESI → HST super-resolution."""

import argparse
import copy
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import FMDataset, denormalize_hst
from models.unet import FMUNet
from models.unet_v2 import UNetV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer updates; useful for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the batch size in the configuration file.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override the DataLoader worker count in the configuration file.")
    return parser.parse_args()


def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def build_source_weight(seg_map, alpha, min_pixels=16):
    """Build per-pixel weight map: 1.0 for background, 1.0+alpha for source pixels.

    Only sources with >= min_pixels are included; smaller ones get weight 1.0.
    """
    weight = torch.ones_like(seg_map, dtype=torch.float32)
    B = seg_map.shape[0]
    for b in range(B):
        seg = seg_map[b, 0]
        labels = seg.unique()
        labels = labels[labels > 0]
        for lab in labels:
            mask_k = seg == lab
            if mask_k.sum() >= min_pixels:
                weight[b, 0][mask_k] = 1.0 + alpha
    return weight


@torch.no_grad()
def validate(model, val_loader, device, data_dir, num_steps=20):
    """Run Euler sampling on validation set. Returns (mse, psnr, flux_err)."""
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_flux_err = 0.0
    count = 0

    for desi, hst_gt, _wht, _seg in val_loader:
        desi = desi.to(device)
        hst_gt = hst_gt.to(device)
        B = desi.shape[0]

        # Euler sampling from noise
        x = torch.randn(B, 1, hst_gt.shape[-2], hst_gt.shape[-1], device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = model(x, t, desi)
            x = x + v * dt

        # MSE & PSNR in normalized [0,1] domain
        pred_np = x.cpu().numpy()
        gt_np = hst_gt.cpu().numpy()
        for j in range(B):
            mse = np.mean((pred_np[j] - gt_np[j]) ** 2)
            total_mse += mse
            total_psnr += 10 * np.log10(1.0 / (mse + 1e-12))

        # Flux error in original domain
        pred_flux = denormalize_hst(pred_np, data_dir)
        gt_flux = denormalize_hst(gt_np, data_dir)
        for j in range(B):
            total_flux_err += abs(pred_flux[j, 0].sum() - gt_flux[j, 0].sum())

        count += B

    return total_mse / count, total_psnr / count, total_flux_err / count


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Shortcuts
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        tcfg["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_dir = tcfg["exp_dir"] + f"_{timestamp}"
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config for reproducibility
    import shutil
    shutil.copy2(args.config, os.path.join(exp_dir, "config.yaml"))

    logger = setup_logging(exp_dir)

    # Seed
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    # Data
    max_ab_ratio = float(dcfg.get("max_ab_ratio", 2.0))
    train_ds = FMDataset(dcfg["data_dir"], split="train", split_file=dcfg["split_file"],
                         max_ab_ratio=max_ab_ratio)
    val_ds = FMDataset(dcfg["data_dir"], split="test", split_file=dcfg["split_file"],
                       max_ab_ratio=max_ab_ratio)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
    )
    logger.info(f"Train samples: {len(train_ds)}, Test samples: {len(val_ds)}")

    # Model
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # EMA
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_decay = tcfg["ema_decay"]
    logger.info(f"EMA decay: {ema_decay}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]),
    )

    # Loss options
    use_wht_weight = tcfg.get("use_wht_weight", False)
    source_weight_alpha = float(tcfg.get("source_weight_alpha", 0.0))
    source_weight_min_pixels = int(tcfg.get("source_weight_min_pixels", 16))
    logger.info(f"use_wht_weight: {use_wht_weight}")
    if source_weight_alpha > 0:
        logger.info(f"  source_weight_alpha: {source_weight_alpha}, min_pixels: {source_weight_min_pixels}")

    total_steps = len(train_loader) * tcfg["epochs"]
    warmup_steps = tcfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info(f"Starting OT-CFM training for {tcfg['epochs']} epochs on {device}")

    reached_max_steps = False
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        t0 = time.time()

        epoch_fm = 0.0

        for i, (desi, hst_gt, wht, seg_map) in enumerate(train_loader):
            desi = desi.to(device)             # (B, 1, 128, 128)
            hst_gt = hst_gt.to(device)         # (B, 1, 512, 512)
            wht = wht.to(device)               # (B, 1, 512, 512)
            seg_map = seg_map.to(device)       # (B, 1, 512, 512) labeled
            B = desi.shape[0]

            # Sample noise and time
            x_0 = torch.randn_like(hst_gt)                    # (B, 1, 512, 512)
            t = torch.rand(B, device=device)                   # (B,) in [0, 1]

            # OT interpolation
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * hst_gt    # (B, 1, 512, 512)

            # Target velocity
            u_t = hst_gt - x_0

            # Predict velocity
            v_pred = model(x_t, t, desi)

            # Velocity residual
            residual_sq = (v_pred - u_t) ** 2

            # Build spatial weight: source regions get higher weight
            if source_weight_alpha > 0:
                sw = build_source_weight(seg_map, source_weight_alpha, source_weight_min_pixels)
            else:
                sw = 1.0

            # Loss: optional WHT weight × optional source weight
            if use_wht_weight:
                w = wht * sw
                loss_fm = (w * residual_sq).sum() / (w.sum() + 1e-8)
            else:
                if source_weight_alpha > 0:
                    loss_fm = (sw * residual_sq).mean() / (sw.mean() + 1e-8)
                else:
                    loss_fm = F.mse_loss(v_pred, u_t)

            loss = loss_fm

            # NaN guard
            if not torch.isfinite(loss):
                logger.info(f"  [WARN] NaN/Inf loss at epoch {epoch+1} iter {i+1}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            # EMA update
            update_ema(ema_model, model, ema_decay)

            epoch_loss += loss.item() * B
            epoch_fm += loss_fm.item() * B
            epoch_count += B

            if args.max_steps is not None and global_step >= args.max_steps:
                reached_max_steps = True
                logger.info(f"Reached max_steps={args.max_steps}; stopping training.")
                break

            if (i + 1) % tcfg["log_every"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch [{epoch+1}/{tcfg['epochs']}] "
                    f"Iter [{i+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}  FM: {loss_fm.item():.6f}  LR: {lr:.2e}"
                )

        avg_loss = epoch_loss / max(epoch_count, 1)
        avg_fm = epoch_fm / max(epoch_count, 1)
        elapsed = time.time() - t0

        epoch_parts = f"Loss: {avg_loss:.6f}  FM: {avg_fm:.6f}"

        # Validation at save epochs
        if (epoch + 1) % tcfg["save_every"] == 0:
            val_mse, val_psnr, val_flux_err = validate(
                ema_model, val_loader, device, dcfg["data_dir"], num_steps=tcfg["val_steps"]
            )
            logger.info(
                f"Epoch [{epoch+1}/{tcfg['epochs']}] "
                f"{epoch_parts}  Val MSE: {val_mse:.2e}  "
                f"Val PSNR: {val_psnr:.2f}dB  Val FluxErr: {val_flux_err:.4f}  "
                f"Time: {elapsed:.1f}s"
            )
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")
        else:
            logger.info(
                f"Epoch [{epoch+1}/{tcfg['epochs']}] "
                f"{epoch_parts}  Time: {elapsed:.1f}s"
            )

        if reached_max_steps:
            break


if __name__ == "__main__":
    main()
