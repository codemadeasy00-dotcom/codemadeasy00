# Cosmic Ray Noise Reduction in Astronomical Images

A hackathon-ready Python project to detect and remove cosmic-ray noise artifacts from astronomical images.

## Project Purpose
Cosmic rays can create sudden bright pixels/streaks in telescope captures. This project provides a simple and understandable pipeline for students to:
- preprocess astronomical images,
- detect likely cosmic-ray artifacts,
- denoise those artifacts,
- and visualize everything through a Streamlit web app.

## Folder Structure

```text
cosmic-ray-noise-reduction/
│
├── backend/
│   ├── preprocessing.py
│   ├── detection.py
│   ├── denoising.py
│   └── pipeline.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── output/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Open terminal in `cosmic-ray-noise-reduction/`.
2. Create and activate virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run frontend/app.py
   ```
5. Upload an astronomical image and view:
   - original image,
   - detected noise mask,
   - denoised output image.

Processed outputs are saved to `data/output/`.

## Notes
- No external APIs are used.
- Uses only open-source Python libraries.
- Designed to be beginner-friendly for 2nd-year B.Tech students.
