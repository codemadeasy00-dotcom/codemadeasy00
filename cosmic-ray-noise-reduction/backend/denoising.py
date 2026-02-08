"""Denoising utilities for removing cosmic-ray artifacts."""

from __future__ import annotations

import cv2
import numpy as np


def remove_cosmic_rays(image: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
    """Remove detected cosmic-ray artifacts with median-based replacement.

    Args:
        image: Normalized grayscale image (float32 in [0, 1]).
        noise_mask: Binary mask where noisy pixels are 1.

    Returns:
        Cleaned normalized grayscale image (float32 in [0, 1]).
    """
    if image.shape != noise_mask.shape:
        raise ValueError("Image and mask must have the same shape.")

    # Create a median-filtered version of the image to estimate local clean signal.
    median_estimate = cv2.medianBlur((image * 255).astype(np.uint8), 5).astype(np.float32) / 255.0

    # Replace only detected noisy pixels with local median values.
    cleaned = image.copy()
    cleaned[noise_mask.astype(bool)] = median_estimate[noise_mask.astype(bool)]
    return cleaned
