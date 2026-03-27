"""
UrbanSAR - Training Script

Training loop for the dual-branch CNN with:
- MSE loss for height regression
- Adam optimizer with configurable LR
- FP16 mixed-precision training
- MAE + RMSE tracking per epoch
- Model checkpointing (save best by validation MAE)
- Early stopping
- CSV logging

Designed to run on Google Colab / Kaggle (auto-detects GPU).

Usage:
    python training/train.py
    python training/train.py --epochs 100 --lr 0.0001 --batch-size 16
"""
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, USE_FP16, VAL_SPLIT, MODEL_DIR, LOGS_DIR,
    SAR_CHANNELS, IMG_SIZE,
)
from models.dual_branch_cnn import DualBranchCNN
from data.dataset import UrbanSARDataset, create_data_splits


def compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: Predicted heights [N, 1] or [N]
        targets: Target heights [N]

    Returns:
        Dict with MAE, RMSE, R² metrics
    """
    preds = predictions.detach().cpu().flatten()
    tgts = targets.detach().cpu().flatten()

    mae = torch.mean(torch.abs(preds - tgts)).item()
    mse = torch.mean((preds - tgts) ** 2).item()
    rmse = np.sqrt(mse)

    # R² score
    ss_res = torch.sum((tgts - preds) ** 2).item()
    ss_tot = torch.sum((tgts - torch.mean(tgts)) ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_fp16: bool = False,
) -> Dict[str, float]:
    """Train for one epoch, return metrics."""
    model.train()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    for sar, optical, heights in dataloader:
        sar = sar.to(device)
        optical = optical.to(device)
        heights = heights.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with autocast():
                predictions = model(sar, optical)
                loss = criterion(predictions, heights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(sar, optical)
            loss = criterion(predictions, heights)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(predictions.detach())
        all_targets.append(heights.detach())

    # Compute epoch metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = round(total_loss / max(num_batches, 1), 4)

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model, return metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    for sar, optical, heights in dataloader:
        sar = sar.to(device)
        optical = optical.to(device)
        heights = heights.to(device).unsqueeze(1)

        predictions = model(sar, optical)
        loss = criterion(predictions, heights)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(predictions)
        all_targets.append(heights)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = round(total_loss / max(num_batches, 1), 4)

    return metrics


def train(
    chip_list: List[Dict],
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    use_fp16: bool = USE_FP16,
    patience: int = EARLY_STOP_PATIENCE,
    save_dir: Path = MODEL_DIR,
    log_dir: Path = LOGS_DIR,
) -> Tuple[nn.Module, Dict]:
    """
    Full training pipeline.

    Args:
        chip_list: List of chip dicts (from select_chips.py)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        use_fp16: Whether to use FP16 mixed precision
        patience: Early stopping patience
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs

    Returns:
        (trained_model, best_metrics)
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data splits
    train_chips, val_chips = create_data_splits(chip_list, val_split=VAL_SPLIT)

    # Datasets
    train_dataset = UrbanSARDataset(train_chips, augment=True, preprocess=True)
    val_dataset = UrbanSARDataset(val_chips, augment=False, preprocess=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = DualBranchCNN(sar_channels=SAR_CHANNELS, pretrained=True)
    model = model.to(device)
    print(model.summary())

    # Loss, optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # FP16 scaler
    scaler = GradScaler() if use_fp16 and device.type == "cuda" else None
    if use_fp16 and device.type != "cuda":
        print("[WARN] FP16 requires CUDA. Falling back to FP32.")
        use_fp16 = False

    # Logging setup
    log_file = log_dir / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_MAE", "train_RMSE", "train_R2",
            "val_loss", "val_MAE", "val_RMSE", "val_R2", "lr", "time_s"
        ])

    # Training loop
    best_val_mae = float("inf")
    best_epoch = 0
    no_improve_count = 0

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Train MAE':>10} | {'Val MAE':>10} | {'Val RMSE':>10} | {'LR':>10}")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_fp16
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t_start

        # Print progress
        print(
            f"{epoch:>6} | {train_metrics['MAE']:>10.4f} | "
            f"{val_metrics['MAE']:>10.4f} | {val_metrics['RMSE']:>10.4f} | "
            f"{current_lr:>10.6f}"
        )

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["MAE"],
                train_metrics["RMSE"], train_metrics["R2"],
                val_metrics["loss"], val_metrics["MAE"],
                val_metrics["RMSE"], val_metrics["R2"],
                current_lr, round(elapsed, 2),
            ])

        # Checkpointing
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_epoch = epoch
            no_improve_count = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": best_val_mae,
                "val_rmse": val_metrics["RMSE"],
                "val_r2": val_metrics["R2"],
                "config": {
                    "sar_channels": SAR_CHANNELS,
                    "img_size": IMG_SIZE,
                    "lr": lr,
                    "batch_size": batch_size,
                },
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"  ✓ Saved best model (MAE: {best_val_mae:.4f})")
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val MAE: {best_val_mae:.4f}")
    print(f"  Model saved: {save_dir / 'best_model.pth'}")
    print(f"  Log saved:   {log_file}")

    # Load best model
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_rmse": checkpoint["val_rmse"],
        "best_val_r2": checkpoint["val_r2"],
    }


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train UrbanSAR dual-branch CNN")
    parser.add_argument("--chips-file", type=str, required=True,
                        help="Path to chips metadata CSV (from select_chips.py)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    # Load chip list from CSV
    import pandas as pd
    df = pd.read_csv(args.chips_file)
    chip_list = []
    for _, row in df.iterrows():
        chip_list.append({
            "sar_path": row["sar_path"],
            "optical_path": row["optical_path"],
            "height_m": row["height_m"],
            "building_id": row.get("building_id", "unknown"),
        })

    print(f"[INFO] Loaded {len(chip_list)} chips from {args.chips_file}")

    model, metrics = train(
        chip_list,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        patience=args.patience,
    )

    print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
