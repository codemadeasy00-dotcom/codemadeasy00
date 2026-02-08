"""End-to-end pipeline for cosmic-ray noise reduction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .denoising import remove_cosmic_rays
from .detection import detect_cosmic_rays
from .preprocessing import preprocess_image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert normalized float image [0,1] to uint8 [0,255]."""
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def run_pipeline(image_path: str | Path, output_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Run preprocessing, detection, and denoising.

    Args:
        image_path: Path to input image.
        output_dir: Directory where output image will be saved.

    Returns:
        original: Preprocessed normalized image.
        mask: Binary noise mask (0 or 1).
        cleaned: Denoised normalized image.
        output_path: Saved output path.
    """
    original = preprocess_image(image_path)
    mask = detect_cosmic_rays(original)
    cleaned = remove_cosmic_rays(original, mask)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(image_path).stem
    output_path = output_dir / f"{input_name}_cleaned.png"
    cv2.imwrite(str(output_path), _to_uint8(cleaned))

    return original, mask, cleaned, output_path
