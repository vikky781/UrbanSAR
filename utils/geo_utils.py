"""
UrbanSAR - Geospatial Utilities

Helper functions for coordinate transforms, building footprint operations,
and mapping predictions to spatial features.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAP_CENTER_LAT, MAP_CENTER_LON


def pixel_to_coords(
    col: float, row: float, transform
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates using rasterio transform.

    Args:
        col: Pixel column (x)
        row: Pixel row (y)
        transform: Rasterio affine transform

    Returns:
        (longitude, latitude)
    """
    x, y = transform * (col, row)
    return float(x), float(y)


def coords_to_pixel(
    lon: float, lat: float, transform
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.

    Args:
        lon: Longitude
        lat: Latitude
        transform: Rasterio affine transform

    Returns:
        (col, row) pixel coordinates
    """
    inv_transform = ~transform
    col, row = inv_transform * (lon, lat)
    return int(col), int(row)


def load_building_footprints(geojson_path: str | Path) -> gpd.GeoDataFrame:
    """
    Load building footprints from GeoJSON.

    Args:
        geojson_path: Path to GeoJSON file

    Returns:
        GeoDataFrame with building geometries
    """
    gdf = gpd.read_file(geojson_path)

    # Ensure CRS is WGS84 for web mapping
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def assign_heights_to_footprints(
    footprints: gpd.GeoDataFrame,
    predictions: List[Dict],
) -> gpd.GeoDataFrame:
    """
    Map predicted heights to building footprint polygons.

    Args:
        footprints: GeoDataFrame with building geometries
        predictions: List of dicts with 'building_id' and 'predicted_height_m'

    Returns:
        GeoDataFrame with added prediction columns
    """
    pred_lookup = {p["building_id"]: p for p in predictions}

    heights = []
    tiers = []
    colors = []

    from models.vulnerability import classify_vulnerability

    for _, row in footprints.iterrows():
        bid = row.get("building_id", row.name)
        if bid in pred_lookup:
            h = pred_lookup[bid]["predicted_height_m"]
            vuln = classify_vulnerability(h)
            heights.append(h)
            tiers.append(vuln["tier"])
            colors.append(vuln["color"])
        else:
            heights.append(None)
            tiers.append("Unknown")
            colors.append("#888888")

    footprints = footprints.copy()
    footprints["predicted_height_m"] = heights
    footprints["vulnerability_tier"] = tiers
    footprints["tier_color"] = colors

    return footprints


def get_bbox_from_footprints(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """
    Get bounding box from building footprints.

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return tuple(bounds)


def create_demo_footprints(n_buildings: int = 10) -> gpd.GeoDataFrame:
    """
    Create demo building footprints for testing the dashboard
    when real data isn't available.

    Generates rectangular buildings around Rotterdam (SpaceNet-6 area).

    Args:
        n_buildings: Number of demo buildings

    Returns:
        GeoDataFrame with demo building polygons
    """
    rng = np.random.RandomState(42)

    buildings = []
    for i in range(n_buildings):
        # Random offset from center
        lat_offset = rng.uniform(-0.01, 0.01)
        lon_offset = rng.uniform(-0.01, 0.01)

        center_lat = MAP_CENTER_LAT + lat_offset
        center_lon = MAP_CENTER_LON + lon_offset

        # Random building size
        w = rng.uniform(0.0002, 0.0005)
        h = rng.uniform(0.0002, 0.0005)

        geometry = box(
            center_lon - w/2, center_lat - h/2,
            center_lon + w/2, center_lat + h/2,
        )

        buildings.append({
            "building_id": f"demo_bldg_{i+1}",
            "geometry": geometry,
            "height_m": rng.uniform(3, 60),  # 3-60m range
        })

    gdf = gpd.GeoDataFrame(buildings, crs="EPSG:4326")
    return gdf
