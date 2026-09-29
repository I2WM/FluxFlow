"""UNet v2 with GroupNorm and ResBlocks for OT-CFM velocity prediction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels):
    groups = 8
    if channels < groups:
        groups = 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetV2(nn.Module):
    """
    UNet v2 for flow matching velocity prediction.

    Uses GroupNorm + ResBlocks instead of LeakyReLU ConvBlocks.

    Input:  x_t (B,1,512,512) + DESI condition (B,1,128,128) + time t (B,)
    Output: velocity v (B,1,512,512)
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(in_ch if j == 0 else ch, ch, time_dim))
            self.down_blocks.append(blocks)
            in_ch = ch
            if i != len(channels) - 1:
                self.downsamples.append(Downsample(in_ch))

        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_block = in_ch + ch if j == 0 else ch
                blocks.append(ResBlock(in_block, ch, time_dim))
            self.up_blocks.append(blocks)
            in_ch = ch
            if i != len(channels) - 1:
                self.upsamples.append(Upsample(in_ch))

        self.final_norm = _group_norm(in_ch)
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

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
        t_emb = self.time_embed(t)

        h = self.init_conv(inp)

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        h = self.final_conv(F.silu(self.final_norm(h)))
        return h
