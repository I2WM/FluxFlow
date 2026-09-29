"""Dataset for OT-CFM DESI → HST super-resolution."""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class FMDataset(Dataset):
    """
    Loads DESI/HST pairs with min-max normalization to [0, 1].

    Returns:
        desi: (1, 128, 128) normalized DESI image
        hst:  (1, 512, 512) normalized HST image
        wht:  (1, 512, 512) WHT weight map normalized to [0, 1].
              Loaded from hst_wht.npy if available, otherwise all ones.
        seg:  (1, 512, 512) labeled segmentation map (0=background, >0=source ID).
              Loaded from hst_masks.npy if available, otherwise all zeros.
    """

    def __init__(self, data_dir, split="train", split_file="sub_train_test_split.json",
                 max_ab_ratio=2.0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_ab_ratio = max_ab_ratio

        # Load split
        sf = self.data_dir / split_file
        with open(sf, "r") as f:
            splits = json.load(f)
        self.sample_names = sorted(splits[split])

        # Load normalization stats
        with open(self.data_dir / "normalize.json", "r") as f:
            stats = json.load(f)
        self.desi_min = stats["desi_sci"]["min"]
        self.desi_max = stats["desi_sci"]["max"]
        self.hst_min = stats["hst_sci"]["min"]
        self.hst_max = stats["hst_sci"]["max"]

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def _clean(img):
        """Replace NaN/Inf: NaN→0, -Inf→0, +Inf→valid_max."""
        finite_pixels = img[np.isfinite(img)]
        valid_max = finite_pixels.max() if finite_pixels.size > 0 else 65535.0
        img = np.nan_to_num(img, nan=0.0, posinf=valid_max, neginf=0.0)
        return img

    def _normalize(self, img, vmin, vmax):
        """Clip and scale to [0, 1]."""
        img = np.clip(img, vmin, vmax)
        return (img - vmin) / (vmax - vmin + 1e-12)

    @staticmethod
    def _normalize_wht(wht):
        """Clean NaN/Inf and normalize WHT per image so mean == 1 (preserves relative inverse-variance)."""
        wht = np.nan_to_num(wht, nan=0.0, posinf=0.0, neginf=0.0)
        wht = np.clip(wht, 0, None)
        m = wht.mean()
        if m > 0:
            wht = wht / m
        else:
            wht = np.ones_like(wht)  # fallback: uniform weight
        return wht

    def __getitem__(self, idx):
        name = self.sample_names[idx]
        folder = self.data_dir / name

        desi = self._clean(np.load(folder / "desi_sci.npy").astype(np.float32))
        hst = self._clean(np.load(folder / "hst_sci.npy").astype(np.float32))

        # Load and normalize WHT map
        wht_path = folder / "hst_wht.npy"
        if wht_path.exists():
            wht = np.load(wht_path).astype(np.float32)
            wht = self._normalize_wht(wht)
        else:
            wht = np.ones_like(hst)  # fallback: uniform weight

        # Load labeled segmentation map (0=background, >0=source ID)
        # Filter out sources with extreme ellipticity (a/b > max_ab_ratio)
        mask_path = folder / "hst_masks.npy"
        if mask_path.exists():
            seg = np.load(mask_path).astype(np.float32)
            src_path = folder / "hst_sources.npz"
            if src_path.exists() and self.max_ab_ratio > 0:
                src = np.load(src_path)
                ab_ratio = src["a"] / (src["b"] + 1e-8)
                bad_ids = src["id"][ab_ratio > self.max_ab_ratio].astype(int)
                for bid in bad_ids:
                    seg[seg == bid] = 0
        else:
            seg = np.zeros_like(hst)

        # Min-max normalize to [0, 1]
        desi = self._normalize(desi, self.desi_min, self.desi_max)
        hst = self._normalize(hst, self.hst_min, self.hst_max)

        # (H, W) → (1, H, W)
        desi = torch.from_numpy(desi).unsqueeze(0)
        hst = torch.from_numpy(hst).unsqueeze(0)
        wht = torch.from_numpy(wht).unsqueeze(0)
        seg = torch.from_numpy(seg).unsqueeze(0)

        return desi, hst, wht, seg


def load_sources(data_dir, name):
    """Load source table from hst_sources.npz.

    Returns dict with keys: x, y, a, b, theta, npix, flag (numpy arrays),
    or None if file doesn't exist.
    """
    path = Path(data_dir) / name / "hst_sources.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def denormalize_hst(img_norm, data_dir):
    """Inverse min-max: [0,1] → original flux domain."""
    with open(Path(data_dir) / "normalize.json", "r") as f:
        stats = json.load(f)
    vmin = stats["hst_sci"]["min"]
    vmax = stats["hst_sci"]["max"]
    return img_norm * (vmax - vmin) + vmin
