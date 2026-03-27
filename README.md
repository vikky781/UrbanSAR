# UrbanSAR

**Multi-Sensor Fusion for Autonomous Building Height Estimation & Disaster Vulnerability Mapping in Smart Cities**

Team Binary Blackhole | Cosmix Abhisarga 2026 | Track 3: Multi-Sensor Fusion | IIIT Sri City

---

## Overview

UrbanSAR fuses Synthetic Aperture Radar (SAR) and optical satellite imagery to autonomously estimate building heights and classify disaster vulnerability — optimized for edge deployment and real-time disaster response.

**Pipeline:** Multi-Sensor Fusion Engine → Height Estimation & Classification → Interactive Smart City Dashboard

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SpaceNet-6 Data

**Primary (AWS S3):**
```bash
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/ ./data/raw/ --recursive --no-sign-request
```

> ⚠️ If `--no-sign-request` fails with "Access Denied":
> 1. Create free AWS account: https://aws.amazon.com/
> 2. Run `aws configure`
> 3. Retry without `--no-sign-request`

**Or use the download script:**
```bash
python data/download_data.py
```

### 3. Select & Crop Building Chips

```bash
python scripts/select_chips.py --num-chips 200
```

### 4. Train the Model

**On Kaggle / Google Colab (GPU required):**
```bash
python training/train.py --chips-file data/processed/chips/chips_metadata.csv --epochs 50
```

### 5. Generate Fallback Data

```bash
# Demo data (for testing dashboard without trained model)
python scripts/generate_fallbacks.py

# From trained model
python scripts/generate_fallbacks.py --from-model
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
UrbanSAR/
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── data/
│   ├── download_data.py       # SpaceNet-6 download (AWS S3 / Kaggle)
│   ├── loader.py              # GeoTIFF + label loading
│   ├── preprocessing.py       # Lee speckle filter, radiometric norm
│   └── dataset.py             # PyTorch Dataset (dual-input)
├── models/
│   ├── dual_branch_cnn.py     # ResNet18 dual-branch fusion model
│   ├── vulnerability.py       # Height → vulnerability tier classifier
│   └── shadow_fallback.py     # SAR shadow geometry fallback (Plan B)
├── training/
│   ├── train.py               # Training loop (FP16, checkpointing)
│   └── evaluate.py            # Metrics + comparison tables
├── dashboard/
│   ├── app.py                 # Streamlit main app (4 pages)
│   ├── map_view.py            # Folium interactive map
│   └── charts.py              # Plotly visualizations
├── utils/
│   ├── fallback.py            # JSON fallback save/load
│   └── geo_utils.py           # Geospatial helpers
├── scripts/
│   ├── select_chips.py        # Chip selection + cropping
│   └── generate_fallbacks.py  # Generate demo/model fallback data
├── checkpoints/               # Trained model files
├── fallback_data/             # JSON fallback outputs
└── logs/                      # Training logs
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| **AI/ML** | PyTorch, ResNet18 (fine-tuned), FP16 quantization |
| **Data** | SpaceNet-6, Rasterio, Rioxarray, GeoPandas |
| **Dashboard** | Streamlit, Folium, Plotly |

## Vulnerability Classification

Height-based proxy heuristic for rapid triage:

| Tier | Height Range | Description |
|------|-------------|-------------|
| 🔴 Critical Risk | < 10m (1-3 floors) | Likely submerged in severe flooding |
| 🟡 Moderate Risk | 10-25m (4-8 floors) | Partial flood exposure |
| 🟢 Evacuation Safe | > 25m (9+ floors) | Viable for vertical evacuation |

> **Note:** This is a height-based proxy. Real vulnerability depends on construction type, materials, and local conditions.

## Fallback Strategy

At every phase boundary, prediction outputs are cached as JSON. If any component fails during demo, the dashboard seamlessly loads cached results.

## Contact

**Team Binary Blackhole** — Vikhyat Gupta & Mohit Choudhary  
📧 vikhyatg7@gmail.com
