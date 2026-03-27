"""
UrbanSAR - Chip Selection Script

Selects and crops building chips from SpaceNet-6 900x900px tiles.

Steps:
    1. Scan tiles for SAR + optical + label triplets
    2. Load building footprints with height labels
    3. Build spatial index of tiles → match buildings to overlapping tiles
    4. Crop building regions from matched tiles
    5. Filter for height diversity
    6. Save cropped chips + metadata CSV

Usage:
    python scripts/select_chips.py --num-chips 500
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DIR, OPTICAL_DIR, LABELS_DIR, CHIPS_DIR
from data.loader import load_height_labels, create_chip_pairs


def get_tile_bounds(tile_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Get geographic bounds of a tile. Returns (minx, miny, maxx, maxy) or None."""
    try:
        with rasterio.open(tile_path) as src:
            b = src.bounds
            return (b.left, b.bottom, b.right, b.top)
    except Exception:
        return None


def bounds_overlap(a: Tuple, b: Tuple) -> bool:
    """Check if two bounding boxes overlap. Both are (minx, miny, maxx, maxy)."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def crop_building_from_tile(
    tile_path: Path,
    building_bounds: Tuple[float, float, float, float],
    padding_px: int = 32,
    target_size: int = 256,
) -> Optional[np.ndarray]:
    """
    Crop a building region from a tile.

    Args:
        tile_path: Path to GeoTIFF tile
        building_bounds: (minx, miny, maxx, maxy) in CRS coordinates
        padding_px: Extra pixels around the building footprint
        target_size: Output chip size (square)

    Returns:
        Cropped image array [C, target_size, target_size] or None
    """
    try:
        with rasterio.open(tile_path) as src:
            # Convert building bounds to pixel coordinates
            minx, miny, maxx, maxy = building_bounds
            inv_transform = ~src.transform

            col_min, row_max = inv_transform * (minx, miny)
            col_max, row_min = inv_transform * (maxx, maxy)

            # Ensure proper ordering
            if col_min > col_max:
                col_min, col_max = col_max, col_min
            if row_min > row_max:
                row_min, row_max = row_max, row_min

            # Add padding and clamp to image bounds
            col_start = max(0, int(col_min) - padding_px)
            row_start = max(0, int(row_min) - padding_px)
            col_end = min(src.width, int(col_max) + padding_px)
            row_end = min(src.height, int(row_max) + padding_px)

            # Validate window dimensions
            w = col_end - col_start
            h = row_end - row_start
            if w < 4 or h < 4:
                return None

            window = rasterio.windows.Window(col_start, row_start, w, h)
            chip = src.read(window=window)

            if chip.size == 0 or chip.shape[1] < 4 or chip.shape[2] < 4:
                return None

    except Exception:
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
    buildings: List[Dict],
    num_chips: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Select chips with diverse building heights using stratified sampling.

    Args:
        buildings: List of building dicts with height info
        num_chips: Target number of chips
        seed: Random seed

    Returns:
        Selected subset of buildings
    """
    rng = np.random.RandomState(seed)

    sorted_buildings = sorted(buildings, key=lambda b: b["height_m"])

    if len(sorted_buildings) <= num_chips:
        return sorted_buildings

    # Height bins: [0-5, 5-10, 10-20, 20-30, 30-50, 50+]
    bins = [0, 5, 10, 20, 30, 50, float("inf")]
    binned = {i: [] for i in range(len(bins) - 1)}

    for b in sorted_buildings:
        for i in range(len(bins) - 1):
            if bins[i] <= b["height_m"] < bins[i + 1]:
                binned[i].append(b)
                break

    per_bin = num_chips // len(binned)
    selected = []

    for i, bldgs in binned.items():
        if not bldgs:
            continue
        n_sample = min(len(bldgs), per_bin)
        selected.extend(rng.choice(bldgs, n_sample, replace=False).tolist())

    # Fill remaining
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

    # ── 1. Load label files ──
    labels_dir = Path(args.labels_dir)
    label_files = sorted(labels_dir.glob("*.geojson")) + sorted(labels_dir.glob("*.json"))

    if not label_files:
        print("[ERROR] No label files found. Download labels first.")
        sys.exit(1)

    print(f"[INFO] Loading labels from {len(label_files)} files...")
    all_buildings = []

    for lf in label_files:
        gdf = load_height_labels(lf)
        if "height_m" in gdf.columns and gdf["height_m"].notna().any():
            for _, row in gdf[gdf["height_m"].notna()].iterrows():
                bldg_id = row.get("Building_ID", row.get("building_id", str(row.name)))
                all_buildings.append({
                    "building_id": bldg_id,
                    "height_m": float(row["height_m"]),
                    "geometry": row.geometry,
                    "bounds": row.geometry.bounds,  # (minx, miny, maxx, maxy)
                    "label_file": lf.name,
                })

    print(f"[INFO] Found {len(all_buildings)} buildings with height labels")
    if not all_buildings:
        print("[ERROR] No buildings with valid heights found. Check label column names.")
        sys.exit(1)

    # ── 2. Find tile pairs and build spatial index ──
    pairs = create_chip_pairs(args.sar_dir, args.optical_dir)
    print(f"[INFO] Found {len(pairs)} SAR-optical tile pairs")

    if not pairs:
        print("[ERROR] No matched SAR-optical tile pairs found.")
        sys.exit(1)

    # Get bounds for each tile pair (use SAR bounds as reference)
    print("[INFO] Building spatial index of tiles...")
    tile_bounds = []
    for p in pairs:
        tb = get_tile_bounds(p["sar"])
        if tb is not None:
            tile_bounds.append({"pair": p, "bounds": tb})

    print(f"[INFO] Indexed {len(tile_bounds)} tiles with valid bounds")

    # ── 3. Select diverse subset ──
    selected = select_diverse_chips(all_buildings, num_chips=args.num_chips)
    print(f"[INFO] Selected {len(selected)} diverse buildings")

    # ── 4. Match buildings to tiles and crop ──
    metadata = []
    chip_count = 0
    skip_count = 0

    for i, bldg in enumerate(selected):
        # Find a tile that overlaps this building
        matched_tile = None
        for tb in tile_bounds:
            if bounds_overlap(bldg["bounds"], tb["bounds"]):
                matched_tile = tb["pair"]
                break

        if matched_tile is None:
            skip_count += 1
            continue

        # Crop from matched SAR and optical tiles
        sar_chip = crop_building_from_tile(
            matched_tile["sar"], bldg["bounds"],
            target_size=args.target_size,
        )
        opt_chip = crop_building_from_tile(
            matched_tile["optical"], bldg["bounds"],
            target_size=args.target_size,
        )

        if sar_chip is None or opt_chip is None:
            skip_count += 1
            continue

        # Save chips
        chip_id = f"chip_{chip_count:04d}"
        np.save(sar_chips_dir / f"{chip_id}_sar.npy", sar_chip)
        np.save(optical_chips_dir / f"{chip_id}_optical.npy", opt_chip)

        metadata.append({
            "chip_id": chip_id,
            "sar_path": str(sar_chips_dir / f"{chip_id}_sar.npy"),
            "optical_path": str(optical_chips_dir / f"{chip_id}_optical.npy"),
            "height_m": bldg["height_m"],
            "building_id": bldg["building_id"],
            "source_tile": matched_tile["sar"].name,
        })
        chip_count += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(selected)} buildings ({chip_count} chips saved, {skip_count} skipped)")

    # ── 5. Save metadata ──
    meta_path = output_dir / "chips_metadata.csv"
    pd.DataFrame(metadata).to_csv(meta_path, index=False)

    print(f"\n[OK] Chip selection complete!")
    print(f"  Chips saved: {chip_count}")
    print(f"  Skipped (no overlap / bad crop): {skip_count}")
    print(f"  Metadata: {meta_path}")
    print(f"  SAR chips: {sar_chips_dir}")
    print(f"  Optical chips: {optical_chips_dir}")

    heights = [m["height_m"] for m in metadata]
    if heights:
        print(f"\n  Height distribution:")
        print(f"    Min:    {min(heights):.1f}m")
        print(f"    Max:    {max(heights):.1f}m")
        print(f"    Mean:   {np.mean(heights):.1f}m")
        print(f"    Median: {np.median(heights):.1f}m")


if __name__ == "__main__":
    main()
