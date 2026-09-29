"""UNet with time conditioning for OT-CFM velocity prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding for time t ∈ [0, 1]."""

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,) → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLPProjection(nn.Module):
    """Project time embedding to (scale, shift) for a conv block."""

    def __init__(self, time_dim, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )

    def forward(self, t_emb):
        """t_emb: (B, time_dim) → scale, shift: (B, out_ch, 1, 1)"""
        ss = self.mlp(t_emb)
        scale, shift = ss.chunk(2, dim=-1)
        return scale[:, :, None, None], shift[:, :, None, None]


class ConvBlock(nn.Module):
    """Double conv with time-conditioned scale-shift and residual."""

    def __init__(self, in_ch, out_ch, time_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.time_proj = TimeMLPProjection(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        scale, shift = self.time_proj(t_emb)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        return h + self.skip(x)


class FMUNet(nn.Module):
    """
    UNet for flow matching velocity prediction.

    Input:  x_t (B,1,512,512) + DESI condition (B,1,128,128) + time t (B,)
    Output: velocity v (B,1,512,512)
    """

    def __init__(self, in_channels=2, out_channels=1, base_dim=48, time_dim=256):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        # dims = [48, 96, 192, 384]

        # Time embedding
        self.time_enc = SinusoidalTimeEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, dims[0], time_dim)
        self.enc2 = ConvBlock(dims[0], dims[1], time_dim)
        self.enc3 = ConvBlock(dims[1], dims[2], time_dim)
        self.enc4 = ConvBlock(dims[2], dims[3], time_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(dims[3], dims[3], time_dim)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Upsampling
        self.up4 = nn.ConvTranspose2d(dims[3], dims[3], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)

        # Decoder (skip connection doubles channels)
        self.dec4 = ConvBlock(dims[3] * 2, dims[3], time_dim)
        self.dec3 = ConvBlock(dims[2] * 2, dims[2], time_dim)
        self.dec2 = ConvBlock(dims[1] * 2, dims[1], time_dim)
        self.dec1 = ConvBlock(dims[0] * 2, dims[0], time_dim)

        # Output: velocity prediction
        self.out_conv = nn.Conv2d(dims[0], out_channels, 3, padding=1)

    def forward(self, x_t, t, y):
        """
        x_t: (B, 1, 512, 512) noisy state
        t:   (B,) time in [0, 1]
        y:   (B, 1, 128, 128) DESI input
        """
        # Upsample DESI and concat with x_t
        y_up = F.interpolate(y, size=x_t.shape[-2:], mode="bicubic", align_corners=False)
        inp = torch.cat([x_t, y_up], dim=1)  # (B, 2, 512, 512)

        # Time embedding
        t_emb = self.time_mlp(self.time_enc(t))  # (B, time_dim)

        # Encoder
        e1 = self.enc1(inp, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(e4), t_emb)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out_conv(d1)  # (B, 1, 512, 512)
