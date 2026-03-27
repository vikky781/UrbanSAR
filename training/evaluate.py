"""
UrbanSAR - Model Evaluation

Compute metrics (MAE, RMSE, R²), generate comparison tables,
and create visualizations for predicted vs. reference heights.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, SAR_CHANNELS, IMG_SIZE
from models.dual_branch_cnn import DualBranchCNN
from models.vulnerability import classify_vulnerability


def load_trained_model(
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DualBranchCNN, Dict]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint (default: best_model.pth)
        device: Target device

    Returns:
        (model, checkpoint_info)
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / "best_model.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    sar_channels = config.get("sar_channels", SAR_CHANNELS)

    model = DualBranchCNN(sar_channels=sar_channels, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "val_mae": checkpoint.get("val_mae", "unknown"),
        "val_rmse": checkpoint.get("val_rmse", "unknown"),
        "val_r2": checkpoint.get("val_r2", "unknown"),
    }

    print(f"[INFO] Loaded model from epoch {info['epoch']}, Val MAE: {info['val_mae']}")
    return model, info


@torch.no_grad()
def run_inference(
    model: DualBranchCNN,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Target device

    Returns:
        (predictions_array, targets_array)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_targets = []

    for sar, optical, heights in dataloader:
        sar = sar.to(device)
        optical = optical.to(device)

        predictions = model(sar, optical)
        all_preds.append(predictions.cpu().numpy().flatten())
        all_targets.append(heights.numpy().flatten())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: Predicted heights
        targets: Reference heights

    Returns:
        Dict with MAE, RMSE, R², MedAE metrics
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    median_ae = np.median(abs_errors)

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Median_AE": round(float(median_ae), 4),
        "Max_Error": round(float(np.max(abs_errors)), 4),
        "Num_Samples": len(predictions),
    }


def generate_comparison_table(
    predictions: np.ndarray,
    targets: np.ndarray,
    building_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a comparison table of predicted vs. reference heights.

    Args:
        predictions: Predicted heights
        targets: Reference heights
        building_ids: Optional list of building identifiers

    Returns:
        DataFrame with columns: building_id, predicted_m, reference_m,
        error_m, abs_error_m, vulnerability_tier
    """
    n = len(predictions)

    if building_ids is None:
        building_ids = [f"Building_{i+1}" for i in range(n)]

    rows = []
    for i in range(n):
        vuln = classify_vulnerability(predictions[i])
        rows.append({
            "building_id": building_ids[i],
            "predicted_m": round(float(predictions[i]), 2),
            "reference_m": round(float(targets[i]), 2),
            "error_m": round(float(predictions[i] - targets[i]), 2),
            "abs_error_m": round(float(abs(predictions[i] - targets[i])), 2),
            "vulnerability_tier": vuln["tier"],
            "est_floors": max(1, int(round(predictions[i] / 3.0))),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("abs_error_m", ascending=True).reset_index(drop=True)


def print_evaluation_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    building_ids: Optional[List[str]] = None,
) -> None:
    """Print a formatted evaluation report to console."""
    metrics = compute_metrics(predictions, targets)
    table = generate_comparison_table(predictions, targets, building_ids)

    print("\n" + "=" * 60)
    print("UrbanSAR - Evaluation Report")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  MAE:        {metrics['MAE']:.4f} meters")
    print(f"  RMSE:       {metrics['RMSE']:.4f} meters")
    print(f"  R²:         {metrics['R2']:.4f}")
    print(f"  Median AE:  {metrics['Median_AE']:.4f} meters")
    print(f"  Max Error:  {metrics['Max_Error']:.4f} meters")
    print(f"  Samples:    {metrics['Num_Samples']}")

    print(f"\nPer-Building Comparison (top 10):")
    print(table.head(10).to_string(index=False))

    # Vulnerability distribution
    tier_counts = table["vulnerability_tier"].value_counts()
    print(f"\nVulnerability Distribution:")
    for tier, count in tier_counts.items():
        pct = 100.0 * count / len(table)
        print(f"  {tier}: {count} ({pct:.1f}%)")
