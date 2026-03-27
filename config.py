"""
UrbanSAR - Central Configuration
All hyperparameters, paths, and settings in one place.
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
# SpaceNet-6 actual directory structure
# Primary: train/AOI_11_Rotterdam/... layout
# Fallback: flat SAR/ optical/ labels/ layout
_SN6_ROOT = RAW_DATA_DIR / "train" / "AOI_11_Rotterdam"

# Use SpaceNet-6 layout if it exists, else fallback to flat
SAR_DIR = _SN6_ROOT / "SAR-Intensity" if (_SN6_ROOT / "SAR-Intensity").exists() else RAW_DATA_DIR / "SAR"
OPTICAL_DIR = _SN6_ROOT / "PS-RGB" if (_SN6_ROOT / "PS-RGB").exists() else RAW_DATA_DIR / "optical"
LABELS_DIR = RAW_DATA_DIR / "labels" if (RAW_DATA_DIR / "labels").exists() and any((RAW_DATA_DIR / "labels").iterdir()) else (_SN6_ROOT / "geojson_buildings" if (_SN6_ROOT / "geojson_buildings").exists() else RAW_DATA_DIR / "labels")
PROCESSED_DIR = DATA_DIR / "processed"
CHIPS_DIR = PROCESSED_DIR / "chips"

MODEL_DIR = PROJECT_ROOT / "checkpoints"
FALLBACK_DIR = PROJECT_ROOT / "fallback_data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, SAR_DIR, OPTICAL_DIR, LABELS_DIR,
          PROCESSED_DIR, CHIPS_DIR, MODEL_DIR, FALLBACK_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
# Image dimensions (SpaceNet-6 chips will be resized to this)
IMG_SIZE = 256

# SAR input channels (HH, VV, HV, VH polarizations)
SAR_CHANNELS = 4

# Optical input channels (RGB)
OPTICAL_CHANNELS = 3

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-4

# FP16 mixed precision
USE_FP16 = True

# Train/validation split ratio
VAL_SPLIT = 0.2

# ============================================================
# VULNERABILITY CLASSIFICATION THRESHOLDS
# ============================================================
# Height-based proxy heuristic (NOT ground truth)
# Real vulnerability depends on construction type, materials, etc.
# These thresholds are for rapid triage / first-responder prioritization
CRITICAL_RISK_MAX_HEIGHT = 10.0    # < 10m (~1-3 floors) → Critical Risk
MODERATE_RISK_MAX_HEIGHT = 25.0    # 10-25m (~4-8 floors) → Moderate Risk
# > 25m (~9+ floors) → Evacuation Safe

VULNERABILITY_TIERS = {
    "Critical Risk": {"max_height": CRITICAL_RISK_MAX_HEIGHT, "color": "#FF4444", "description": "Low-rise, likely submerged in severe flooding"},
    "Moderate Risk": {"max_height": MODERATE_RISK_MAX_HEIGHT, "color": "#FFAA00", "description": "Mid-rise, partial flood exposure"},
    "Evacuation Safe": {"max_height": float("inf"), "color": "#44BB44", "description": "High-rise, viable for vertical evacuation"},
}

# ============================================================
# SAR PREPROCESSING
# ============================================================
LEE_FILTER_WINDOW = 7  # Window size for Lee speckle filter

# ============================================================
# SHADOW FALLBACK (Plan B)
# ============================================================
# Default radar incidence angle for SpaceNet-6 (degrees)
# SpaceNet-6 uses Capella SAR with varying incidence angles
# This is a default; actual value should come from metadata
DEFAULT_INCIDENCE_ANGLE_DEG = 40.0

# ============================================================
# DASHBOARD
# ============================================================
# Rotterdam, Netherlands (SpaceNet-6 coverage area)
MAP_CENTER_LAT = 51.9225
MAP_CENTER_LON = 4.47917
MAP_ZOOM = 14

STREAMLIT_PAGE_TITLE = "UrbanSAR - Smart City Dashboard"
STREAMLIT_LAYOUT = "wide"

# ============================================================
# SPACENET-6 AWS S3
# ============================================================
S3_BUCKET = "s3://spacenet-dataset/spacenet/SN6_buildings/"
