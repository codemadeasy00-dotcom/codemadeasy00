"""Cosmic-ray artifact detection utilities."""

from __future__ import annotations

import cv2
import numpy as np


def detect_cosmic_rays(
    image: np.ndarray,
    threshold_quantile: float = 0.995,
    deviation_factor: float = 3.5,
) -> np.ndarray:
    """Detect cosmic-ray pixels using thresholding + local deviation.

    Args:
        image: Normalized grayscale image (float32, range [0, 1]).
        threshold_quantile: High-intensity quantile for bright outliers.
        deviation_factor: Sensitivity multiplier for local deviation.

    Returns:
        Binary mask (uint8) with 1 where cosmic-ray noise is detected.
    """
    if image.ndim != 2:
        raise ValueError("detect_cosmic_rays expects a 2D grayscale image.")

    # 1) Global bright-pixel detection.
    intensity_threshold = float(np.quantile(image, threshold_quantile))
    bright_mask = image >= intensity_threshold

    # 2) Local deviation from neighborhood median.
    local_median = cv2.medianBlur((image * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    local_abs_dev = np.abs(image - local_median)
    mad = np.median(local_abs_dev)
    # Avoid division by zero in extremely smooth images.
    robust_scale = max(mad, 1e-6)
    deviation_mask = local_abs_dev > (deviation_factor * robust_scale)

    # Combine both criteria to reduce false positives.
    combined_mask = bright_mask & deviation_mask

    # Clean isolated noise in the mask with morphology.
    mask_uint8 = (combined_mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    return (cleaned > 0).astype(np.uint8)
