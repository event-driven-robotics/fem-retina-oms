from __future__ import annotations
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import math

def ensure_grayscale_L(img: Image.Image) -> Image.Image:
    """Return an 'L' (grayscale) version of img."""
    if img.mode == 'L':
        return img
    # Convert keeping luminance
    return ImageOps.grayscale(img)

def ensure_rgba(img: Image.Image) -> Image.Image:
    """Ensure the image is RGBA (has alpha)."""
    if img.mode == 'RGBA':
        return img
    return img.convert('RGBA')

def transform_object(obj_rgba: Image.Image,
                     angle_deg: float = 0.0,
                     scale: float = 1.0,
                     resample=Image.BICUBIC) -> Image.Image:
    """Scale and rotate the object RGBA, preserving alpha.
    Returns a new RGBA image possibly with changed size due to rotation (expand=True).
    """
    if scale != 1.0:
        w, h = obj_rgba.size
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        obj_rgba = obj_rgba.resize(new_size, resample=resample)
    if angle_deg % 360 != 0:
        obj_rgba = obj_rgba.rotate(angle_deg, resample=resample, expand=True)
    return obj_rgba

def make_mask(base_size: Tuple[int, int],
              obj_rgba: Image.Image,
              xy: Tuple[int, int],
              mode: str = 'binary',
              thresh: int = 1) -> Image.Image:
    """Create a mask image ('L') from the transformed object's alpha placed at xy.
    mode='binary' uses alpha > thresh -> 255 else 0.
    mode='soft' uses the alpha directly (0..255).
    """
    W, H = base_size
    mask = Image.new('L', (W, H), 0)
    alpha = obj_rgba.split()[-1]

    if mode == 'soft':
        mask.alpha_composite(Image.merge('RGBA', (alpha, alpha, alpha, alpha)), dest=xy)
        mask = mask.split()[0]
        return mask

    # Binary
    bin_alpha = alpha.point(lambda a: 255 if a > thresh else 0, mode='L')
    mask.paste(bin_alpha, box=xy)
    return mask

def combine_masks(mask1: Optional[Image.Image], mask2: Optional[Image.Image]) -> Optional[Image.Image]:
    if mask1 is None and mask2 is None:
        return None
    if mask1 is None:
        return mask2.copy()
    if mask2 is None:
        return mask1.copy()
    a = np.asarray(mask1, dtype=np.uint8)
    b = np.asarray(mask2, dtype=np.uint8)
    c = np.maximum(a, b)
    return Image.fromarray(c, mode='L')

def alpha_area(obj_rgba: Image.Image) -> float:
    """Effective object area in pixels using soft alpha (sum(alpha)/255).
    Counts fractional coverage at edges for anti-aliased PNGs.
    """
    a = np.asarray(obj_rgba.split()[-1], dtype=np.float32)
    return float(a.sum() / 255.0)
