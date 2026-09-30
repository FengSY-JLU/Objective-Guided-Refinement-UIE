from pathlib import Path

import numpy as np
import torch
from PIL import Image


def srgb_to_linear(x):
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp_min(0.0) ** 2.4,
    )


def linear_to_srgb(x):
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.clamp_min(1e-8) ** (1.0 / 2.4) - 0.055,
    )


def load_rgb(path, device=None):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = srgb_to_linear(x).clamp(0.0, 1.0)
    if device is not None:
        x = x.to(device)
    return x


def save_rgb(path, x_linear):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x_linear.detach().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    x = linear_to_srgb(x).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
