"""Preprocessing utilities for astronomical images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load an image, convert to grayscale, and normalize pixel values.

    Args:
        image_path: Path to an input image.

    Returns:
        A normalized grayscale image as float32 NumPy array in [0, 1].

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV fails to load the image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert image to grayscale if needed.
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize to [0, 1] for stable downstream processing.
    normalized = cv2.normalize(
        gray.astype(np.float32), None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX
    )
    return normalized
