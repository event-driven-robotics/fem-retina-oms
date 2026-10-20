"""Load a DOVES image and raw gaze samples; render a background-only sequence."""
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


def load_doves(image_path, recording_path, subject, image_shape=(768, 1024),
               contrast_boost=False):
    """Return a uint8 image and (2, N) gaze array using a zero-based subject index.

    Supply both filenames explicitly, as they appear in the downloaded archive.
    No resampling, missing-data interpolation, or saccade filtering is performed.
    """
    pixels = np.fromfile(image_path, dtype=">u2")
    if pixels.size != np.prod(image_shape):
        raise ValueError(f"Expected {np.prod(image_shape)} big-endian uint16 pixels")
    image = pixels.reshape(image_shape).astype(np.float32)
    if image.max() <= 0:
        raise ValueError("DOVES image is empty")
    image /= image.max()
    if contrast_boost:
        image = np.clip(image * 1.3 + .05, 0, 1)
    data = loadmat(recording_path)["eye_data"]
    if not isinstance(subject, (int, np.integer)) or not 0 <= subject < data.shape[1]:
        raise ValueError("subject must be a valid zero-based eye_data index")
    eye = np.asarray(data[0, subject], dtype=float)
    if eye.ndim != 2 or eye.shape[0] < 2 or eye.shape[1] < 3:
        raise ValueError("eye_data entry must have two rows and at least three samples")
    if not np.isfinite(eye[:2]).all():
        raise ValueError("Non-finite gaze samples: apply the study's missing-data policy explicitly")
    return (image * 255).astype(np.uint8), eye[:2]


def render_background(image, trace, frames_dir, crop_shape=(720, 960)):
    """Translate a static image with (N, 2) offsets, then center-crop each frame."""
    image, trace = np.asarray(image), np.asarray(trace)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image must be a grayscale uint8 array")
    if trace.ndim != 2 or trace.shape[1] != 2 or not len(trace) or not np.isfinite(trace).all():
        raise ValueError("trace must be a nonempty, finite (N, 2) array")
    h, w = crop_shape
    if not 0 < h <= image.shape[0] or not 0 < w <= image.shape[1]:
        raise ValueError("crop_shape must fit the input image")
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=False)
    y0, x0 = (image.shape[0] - h)//2, (image.shape[1] - w)//2
    for i, (x, y) in enumerate(trace):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, y]]), image.shape[::-1])
        if not cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), shifted[y0:y0+h, x0:x0+w]):
            raise OSError("Failed to write frame")
