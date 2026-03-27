"""
UrbanSAR - Data Loader

Functions to load SAR GeoTIFFs, optical GeoTIFFs, and 3DBAG height labels
from the SpaceNet-6 dataset.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DIR, OPTICAL_DIR, LABELS_DIR


def load_sar_image(path: str | Path) -> Tuple[np.ndarray, dict]:
    """
    Load a SAR GeoTIFF image.

    SpaceNet-6 SAR has up to 4 polarization channels (HH, VV, HV, VH).

    Args:
        path: Path to SAR GeoTIFF file

    Returns:
        Tuple of (image_array [C, H, W], metadata_dict)
        where C is the number of polarization channels
    """
    path = Path(path)
    with rasterio.open(path) as src:
        image = src.read()  # Shape: (bands, height, width)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }
    return image.astype(np.float32), meta


def load_optical_image(path: str | Path) -> Tuple[np.ndarray, dict]:
    """
    Load an optical (PS-RGB) GeoTIFF image.

    SpaceNet-6 optical images are typically 3-band RGB.

    Args:
        path: Path to optical GeoTIFF file

    Returns:
        Tuple of (image_array [C, H, W], metadata_dict)
    """
    path = Path(path)
    with rasterio.open(path) as src:
        image = src.read()  # Shape: (bands, height, width)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }
    return image.astype(np.float32), meta


def load_height_labels(geojson_path: str | Path) -> gpd.GeoDataFrame:
    """
    Load building footprints with 3DBAG-derived height labels from GeoJSON.

    SpaceNet-6 building annotations include height information sourced
    from the 3D Basisregistratie Adressen en Gebouwen (3DBAG) dataset,
    derived from aerial LiDAR.

    The height fields may include:
        - 'height_m' or 'height'
        - '75p_mean' (75th percentile mean height)
        - 'median_height'

    Args:
        geojson_path: Path to GeoJSON annotation file

    Returns:
        GeoDataFrame with building footprints and height labels
    """
    gdf = gpd.read_file(geojson_path)

    # Identify the height column (SpaceNet-6 may use different field names)
    height_column = None
    possible_names = [
        "height_m", "height", "Height", "HEIGHT",
        "roof_075mean", "roof_075median", "roof_075stdev",
        "75p_mean", "median_height", "mean_height",
        "bldg_height", "building_height",
        "height_estimate", "ht_agl",
    ]
    for col in possible_names:
        if col in gdf.columns:
            height_column = col
            break

    if height_column is None:
        # Check for any column with 'height' in the name
        for col in gdf.columns:
            if "height" in col.lower() or "ht" in col.lower():
                height_column = col
                break

    if height_column is not None:
        # Standardize to 'height_m'
        gdf["height_m"] = gdf[height_column].astype(float)
        # Remove invalid heights
        gdf = gdf[gdf["height_m"] > 0].reset_index(drop=True)
        print(f"[INFO] Loaded {len(gdf)} buildings with height label '{height_column}'")
    else:
        print(f"[WARN] No height column found. Available columns: {list(gdf.columns)}")
        gdf["height_m"] = np.nan

    return gdf


def create_chip_pairs(
    sar_dir: str | Path = SAR_DIR,
    optical_dir: str | Path = OPTICAL_DIR,
    label_path: Optional[str | Path] = None,
) -> List[Dict]:
    """
    Create matched SAR + optical + label triplets.

    Matches files by common identifiers in filenames (tile ID / chip ID).

    Args:
        sar_dir: Directory containing SAR GeoTIFFs
        optical_dir: Directory containing optical GeoTIFFs
        label_path: Path to GeoJSON with building annotations

    Returns:
        List of dicts: [{"sar": path, "optical": path, "tile_id": id, "labels": GeoDataFrame}, ...]
    """
    sar_dir = Path(sar_dir)
    optical_dir = Path(optical_dir)

    # Collect all SAR and optical files
    sar_files = sorted(list(sar_dir.glob("*.tif")) + list(sar_dir.glob("*.tiff")))
    optical_files = sorted(list(optical_dir.glob("*.tif")) + list(optical_dir.glob("*.tiff")))

    # Build lookup by tile ID (extract numeric/common portion from filename)
    def extract_tile_id(filename: str) -> str:
        """Extract tile identifier from SpaceNet-6 filename."""
        # SpaceNet-6 naming: SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190823111610_...
        # We extract the unique tile portion
        parts = filename.replace(".tif", "").replace(".tiff", "").split("_")
        # Try to find a numeric tile ID or use last meaningful part
        for part in reversed(parts):
            if part.isdigit() and len(part) >= 3:
                return part
        # Fallback: use everything after the modality indicator
        return "_".join(parts[-2:]) if len(parts) > 2 else filename

    sar_lookup = {}
    for f in sar_files:
        tile_id = extract_tile_id(f.name)
        sar_lookup[tile_id] = f

    optical_lookup = {}
    for f in optical_files:
        tile_id = extract_tile_id(f.name)
        optical_lookup[tile_id] = f

    # Load labels if provided
    labels_gdf = None
    if label_path:
        labels_gdf = load_height_labels(label_path)

    # Match pairs
    pairs = []
    common_ids = set(sar_lookup.keys()) & set(optical_lookup.keys())

    for tile_id in sorted(common_ids):
        pair = {
            "tile_id": tile_id,
            "sar": sar_lookup[tile_id],
            "optical": optical_lookup[tile_id],
            "labels": labels_gdf,  # Same labels GDF for all (spatially filtered later)
        }
        pairs.append(pair)

    print(f"[INFO] Found {len(pairs)} matched SAR-optical pairs")
    print(f"  SAR files: {len(sar_files)}, Optical files: {len(optical_files)}")
    if len(common_ids) < min(len(sar_files), len(optical_files)):
        print(f"  [WARN] {min(len(sar_files), len(optical_files)) - len(common_ids)} files had no match")

    return pairs
