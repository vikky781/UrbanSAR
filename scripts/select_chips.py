"""
UrbanSAR - Chip Selection Script

Selects and crops 150-200 building chips from SpaceNet-6 900x900px tiles.

Steps:
    1. Scan tiles for SAR + optical + label triplets
    2. Load building footprints with height labels
    3. Crop building regions from tiles using footprint bounding boxes
    4. Filter for height diversity
    5. Save cropped chips + metadata CSV

Usage:
    python scripts/select_chips.py --data-dir data/raw --output-dir data/processed/chips --num-chips 200
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DIR, OPTICAL_DIR, LABELS_DIR, CHIPS_DIR
from data.loader import load_height_labels, create_chip_pairs


def crop_building_from_tile(
    tile_path: Path,
    building_bounds: Tuple[float, float, float, float],
    padding_px: int = 32,
    target_size: int = 256,
) -> np.ndarray:
    """
    Crop a building region from a 900x900 tile.

    Args:
        tile_path: Path to GeoTIFF tile
        building_bounds: (minx, miny, maxx, maxy) in CRS coordinates
        padding_px: Extra pixels around the building footprint
        target_size: Output chip size (square)

    Returns:
        Cropped image array [C, target_size, target_size]
    """
    with rasterio.open(tile_path) as src:
        # Get window from geographic bounds
        try:
            window = from_bounds(*building_bounds, transform=src.transform)
        except Exception:
            # If bounds don't intersect, return None
            return None

        # Add padding
        row_start = max(0, int(window.row_off) - padding_px)
        col_start = max(0, int(window.col_off) - padding_px)
        row_end = min(src.height, int(window.row_off + window.height) + padding_px)
        col_end = min(src.width, int(window.col_off + window.width) + padding_px)

        # Read windowed region
        window_read = rasterio.windows.Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start
        )

        try:
            chip = src.read(window=window_read)
        except Exception:
            return None

    if chip.size == 0:
        return None

    # Resize to target size
    from PIL import Image
    channels = chip.shape[0]
    resized = np.zeros((channels, target_size, target_size), dtype=np.float32)
    for c in range(channels):
        pil_img = Image.fromarray(chip[c].astype(np.float32))
        pil_resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
        resized[c] = np.array(pil_resized)

    return resized


def select_diverse_chips(
    buildings_with_tiles: List[Dict],
    num_chips: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Select chips with diverse building heights.

    Strategy: bin heights into ranges, sample evenly from each bin
    to ensure training data covers low-rise, mid-rise, and high-rise.

    Args:
        buildings_with_tiles: List of building dicts with height info
        num_chips: Target number of chips
        seed: Random seed

    Returns:
        Selected subset of buildings
    """
    rng = np.random.RandomState(seed)

    # Sort by height
    sorted_buildings = sorted(buildings_with_tiles, key=lambda b: b["height_m"])

    if len(sorted_buildings) <= num_chips:
        return sorted_buildings

    # Create height bins: [0-5, 5-10, 10-20, 20-30, 30-50, 50+]
    bins = [0, 5, 10, 20, 30, 50, float("inf")]
    binned = {i: [] for i in range(len(bins) - 1)}

    for b in sorted_buildings:
        for i in range(len(bins) - 1):
            if bins[i] <= b["height_m"] < bins[i + 1]:
                binned[i].append(b)
                break

    # Sample evenly from each bin
    per_bin = num_chips // len(binned)
    selected = []

    for i, buildings in binned.items():
        if len(buildings) == 0:
            continue
        n_sample = min(len(buildings), per_bin)
        selected.extend(rng.choice(buildings, n_sample, replace=False).tolist())

    # Fill remaining slots from largest bin
    if len(selected) < num_chips:
        remaining = [b for b in sorted_buildings if b not in selected]
        n_more = min(len(remaining), num_chips - len(selected))
        if n_more > 0:
            selected.extend(rng.choice(remaining, n_more, replace=False).tolist())

    return selected[:num_chips]


def main():
    parser = argparse.ArgumentParser(description="Select and crop building chips")
    parser.add_argument("--sar-dir", type=str, default=str(SAR_DIR))
    parser.add_argument("--optical-dir", type=str, default=str(OPTICAL_DIR))
    parser.add_argument("--labels-dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(CHIPS_DIR))
    parser.add_argument("--num-chips", type=int, default=200)
    parser.add_argument("--target-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sar_chips_dir = output_dir / "SAR"
    optical_chips_dir = output_dir / "optical"
    sar_chips_dir.mkdir(parents=True, exist_ok=True)
    optical_chips_dir.mkdir(parents=True, exist_ok=True)

    # Find label files
    labels_dir = Path(args.labels_dir)
    label_files = list(labels_dir.glob("*.geojson")) + list(labels_dir.glob("*.json"))

    if not label_files:
        print("[ERROR] No label files found. Run download_data.py first.")
        sys.exit(1)

    # Load all labels
    print(f"[INFO] Loading labels from {len(label_files)} files...")
    all_buildings = []

    for lf in label_files:
        gdf = load_height_labels(lf)
        if "height_m" in gdf.columns and gdf["height_m"].notna().any():
            for _, row in gdf[gdf["height_m"].notna()].iterrows():
                all_buildings.append({
                    "building_id": row.get("building_id", row.name),
                    "height_m": row["height_m"],
                    "geometry": row.geometry,
                    "bounds": row.geometry.bounds,
                    "label_file": lf.name,
                })

    print(f"[INFO] Found {len(all_buildings)} buildings with height labels")

    # Find matching tiles
    pairs = create_chip_pairs(args.sar_dir, args.optical_dir)
    print(f"[INFO] Found {len(pairs)} SAR-optical tile pairs")

    # Select diverse subset
    selected = select_diverse_chips(all_buildings, num_chips=args.num_chips)
    print(f"[INFO] Selected {len(selected)} diverse buildings")

    # Crop chips
    metadata = []
    chip_count = 0

    for i, bldg in enumerate(selected):
        # Find the best matching tile for this building
        # (simplified: use first tile - in practice, spatially match)
        if not pairs:
            print("[ERROR] No tile pairs available for cropping.")
            break

        sar_tile = pairs[min(i % len(pairs), len(pairs) - 1)]["sar"]
        opt_tile = pairs[min(i % len(pairs), len(pairs) - 1)]["optical"]

        sar_chip = crop_building_from_tile(
            sar_tile, bldg["bounds"],
            target_size=args.target_size,
        )
        opt_chip = crop_building_from_tile(
            opt_tile, bldg["bounds"],
            target_size=args.target_size,
        )

        if sar_chip is None or opt_chip is None:
            continue

        # Save chips as numpy files
        chip_id = f"chip_{chip_count:04d}"
        np.save(sar_chips_dir / f"{chip_id}_sar.npy", sar_chip)
        np.save(optical_chips_dir / f"{chip_id}_optical.npy", opt_chip)

        metadata.append({
            "chip_id": chip_id,
            "sar_path": str(sar_chips_dir / f"{chip_id}_sar.npy"),
            "optical_path": str(optical_chips_dir / f"{chip_id}_optical.npy"),
            "height_m": bldg["height_m"],
            "building_id": bldg["building_id"],
            "source_tile": sar_tile.name,
        })
        chip_count += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(selected)} buildings ({chip_count} chips saved)")

    # Save metadata CSV
    meta_path = output_dir / "chips_metadata.csv"
    pd.DataFrame(metadata).to_csv(meta_path, index=False)

    print(f"\n[OK] Chip selection complete!")
    print(f"  Chips saved: {chip_count}")
    print(f"  Metadata: {meta_path}")
    print(f"  SAR chips: {sar_chips_dir}")
    print(f"  Optical chips: {optical_chips_dir}")

    # Height distribution summary
    heights = [m["height_m"] for m in metadata]
    if heights:
        print(f"\n  Height distribution:")
        print(f"    Min:    {min(heights):.1f}m")
        print(f"    Max:    {max(heights):.1f}m")
        print(f"    Mean:   {np.mean(heights):.1f}m")
        print(f"    Median: {np.median(heights):.1f}m")


if __name__ == "__main__":
    main()
