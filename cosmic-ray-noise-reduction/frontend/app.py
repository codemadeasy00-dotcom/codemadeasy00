"""Streamlit app for visualizing cosmic-ray noise reduction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

# Allow importing backend modules when running from frontend directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="Cosmic Ray Noise Reduction", layout="wide")
st.title("🌌 Cosmic Ray Noise Reduction in Astronomical Images")
st.write(
    "Upload an astronomical image, detect cosmic-ray artifacts (bright outliers), "
    "and view the denoised result produced by median-based correction."
)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded_file is not None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    out_dir = PROJECT_ROOT / "data" / "output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = Path(tmp_file.name)

    saved_input_path = raw_dir / uploaded_file.name
    saved_input_path.write_bytes(temp_path.read_bytes())

    original, mask, cleaned, output_path = run_pipeline(saved_input_path, out_dir)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original (Preprocessed)")
        st.image(original, clamp=True)

    with col2:
        st.subheader("Detected Noise Mask")
        st.image(mask * 255, clamp=True)

    with col3:
        st.subheader("Noise-Reduced Output")
        st.image(cleaned, clamp=True)

    st.success(f"Processed image saved to: {output_path}")
