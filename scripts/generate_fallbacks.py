"""
UrbanSAR - Generate Fallback Data

Runs model inference and saves results as JSON for demo safety.
Also generates demo fallback data for dashboard testing.

Usage:
    python scripts/generate_fallbacks.py                    # Generate demo fallback
    python scripts/generate_fallbacks.py --from-model       # Generate from trained model
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, CHIPS_DIR, FALLBACK_DIR
from utils.fallback import save_fallback
from models.vulnerability import batch_classify


def generate_demo_fallback() -> dict:
    """
    Generate demo prediction data for dashboard testing.
    Simulates realistic building height predictions.
    """
    rng = np.random.RandomState(42)
    n_buildings = 15

    # Realistic Rotterdam building heights
    true_heights = [
        4.5, 7.2, 12.8, 18.5, 22.0,
        6.1, 9.3, 15.6, 28.4, 35.2,
        3.8, 11.0, 20.1, 42.5, 8.7,
    ]

    # Simulated predictions with realistic error
    predictions = [max(2, h + rng.normal(0, 1.5)) for h in true_heights]
    building_ids = [f"ROT_BLDG_{i+1:03d}" for i in range(n_buildings)]

    # Classify
    classifications = batch_classify(predictions)

    # Build complete data structure
    data = {
        "predictions": [round(p, 2) for p in predictions],
        "references": [round(h, 2) for h in true_heights],
        "building_ids": building_ids,
        "classifications": classifications,
        "buildings": [],
    }

    # Per-building details
    for i in range(n_buildings):
        data["buildings"].append({
            "building_id": building_ids[i],
            "predicted_height_m": round(predictions[i], 2),
            "reference_height_m": round(true_heights[i], 2),
            "error_m": round(predictions[i] - true_heights[i], 2),
            "vulnerability_tier": classifications[i]["tier"],
            "est_floors": max(1, int(round(predictions[i] / 3.0))),
            "lat": 51.9225 + rng.uniform(-0.01, 0.01),
            "lon": 4.47917 + rng.uniform(-0.01, 0.01),
        })

    # Summary metrics
    errors = [p - r for p, r in zip(predictions, true_heights)]
    abs_errors = [abs(e) for e in errors]
    data["metrics"] = {
        "MAE": round(float(np.mean(abs_errors)), 4),
        "RMSE": round(float(np.sqrt(np.mean([e**2 for e in errors]))), 4),
        "R2": round(float(1 - sum(e**2 for e in errors) / sum((r - np.mean(true_heights))**2 for r in true_heights)), 4),
        "Num_Samples": n_buildings,
    }

    return data


def generate_model_fallback(chips_file: str, model_path: str = None) -> dict:
    """
    Generate fallback from trained model inference.

    Args:
        chips_file: Path to chips_metadata.csv
        model_path: Path to model checkpoint
    """
    import torch
    from torch.utils.data import DataLoader
    from training.evaluate import load_trained_model, run_inference
    from data.dataset import UrbanSARDataset

    # Load chips
    df = pd.read_csv(chips_file)
    chip_list = []
    for _, row in df.iterrows():
        chip_list.append({
            "sar_path": row["sar_path"],
            "optical_path": row["optical_path"],
            "height_m": row["height_m"],
            "building_id": row.get("building_id", "unknown"),
        })

    # Load model
    if model_path:
        model, info = load_trained_model(model_path)
    else:
        model, info = load_trained_model()

    # Run inference
    dataset = UrbanSARDataset(chip_list, augment=False, preprocess=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    predictions, targets = run_inference(model, loader)

    # Build data
    building_ids = [c["building_id"] for c in chip_list]
    classifications = batch_classify(predictions.tolist())

    errors = predictions - targets
    abs_errors = np.abs(errors)

    data = {
        "predictions": predictions.tolist(),
        "references": targets.tolist(),
        "building_ids": building_ids,
        "classifications": classifications,
        "metrics": {
            "MAE": round(float(np.mean(abs_errors)), 4),
            "RMSE": round(float(np.sqrt(np.mean(errors**2))), 4),
            "R2": round(float(1 - np.sum(errors**2) / np.sum((targets - np.mean(targets))**2)), 4),
            "Num_Samples": len(predictions),
        },
    }

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate fallback data")
    parser.add_argument("--from-model", action="store_true",
                        help="Generate from trained model instead of demo data")
    parser.add_argument("--chips-file", type=str,
                        default=str(CHIPS_DIR / "chips_metadata.csv"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--phase", type=str, default="predictions",
                        help="Phase name for the fallback file")
    args = parser.parse_args()

    if args.from_model:
        print("[INFO] Generating fallback from trained model...")
        data = generate_model_fallback(args.chips_file, args.model_path)
    else:
        print("[INFO] Generating demo fallback data...")
        data = generate_demo_fallback()

    save_fallback(data, args.phase)
    print(f"\n[OK] Fallback saved as '{args.phase}.json'")
    print(f"  Buildings: {data['metrics']['Num_Samples']}")
    print(f"  MAE: {data['metrics']['MAE']:.4f}m")
    print(f"  RMSE: {data['metrics']['RMSE']:.4f}m")


if __name__ == "__main__":
    main()
