"""
UrbanSAR - Shadow-Based Height Estimation Fallback (Plan B)

SAR-specific geometric shadow analysis for building height estimation.
Uses radar geometry (incidence angle), NOT optical sun-based shadows.

When to use this fallback:
    - If the dual-branch CNN fails to converge by Hour 14 of the sprint
    - As a sanity check / secondary estimate alongside CNN predictions
    - When you need a quick estimate without running inference

SAR Shadow Geometry:
    In SAR imagery, buildings cause shadows on the far-range side.
    The shadow appears as a dark (low backscatter) region.
    Shadow length relates to building height via:

        height = shadow_length_ground_range × tan(incidence_angle)

    where:
        - shadow_length_ground_range = shadow_length_pixels × pixel_spacing
        - incidence_angle = radar look angle from vertical (degrees)
        - The incidence angle comes from SAR metadata
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_INCIDENCE_ANGLE_DEG


def estimate_height_from_shadow(
    shadow_length_px: float,
    pixel_spacing_m: float,
    incidence_angle_deg: float = DEFAULT_INCIDENCE_ANGLE_DEG,
) -> float:
    """
    Estimate building height from SAR shadow length using radar geometry.

    Formula:
        ground_range_shadow = shadow_length_px * pixel_spacing_m
        height = ground_range_shadow * tan(incidence_angle)

    Args:
        shadow_length_px: Shadow length in pixels (measured in ground range direction)
        pixel_spacing_m: Ground pixel spacing in meters
        incidence_angle_deg: Radar incidence angle in degrees (from vertical)

    Returns:
        Estimated building height in meters
    """
    # Convert incidence angle to radians
    incidence_rad = np.radians(incidence_angle_deg)

    # Convert pixel shadow length to ground range distance
    shadow_length_ground = shadow_length_px * pixel_spacing_m

    # Height = shadow_length * tan(incidence_angle)
    height = shadow_length_ground * np.tan(incidence_rad)

    return float(max(0.0, height))


def detect_shadows_sar(
    sar_image: np.ndarray,
    threshold_percentile: float = 10.0,
    min_shadow_area_px: int = 20,
) -> np.ndarray:
    """
    Detect shadow regions in SAR image using adaptive thresholding.

    SAR shadows are regions of very low backscatter (dark areas).
    Uses percentile-based thresholding for adaptivity.

    Args:
        sar_image: SAR image [H, W] or [C, H, W] (if multi-channel, uses first channel)
        threshold_percentile: Percentile below which pixels are considered shadow
        min_shadow_area_px: Minimum connected component area to keep

    Returns:
        Binary shadow mask [H, W] (1 = shadow, 0 = non-shadow)
    """
    # Use single channel for shadow detection
    if sar_image.ndim == 3:
        img = sar_image[0]  # Use first polarization (HH typically)
    else:
        img = sar_image

    img = img.astype(np.float64)

    # Adaptive threshold based on percentile
    threshold = np.percentile(img[img > 0], threshold_percentile)
    shadow_mask = (img < threshold) & (img > 0)

    # Remove small components (noise)
    labeled, num_features = ndimage.label(shadow_mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if np.sum(component) < min_shadow_area_px:
            shadow_mask[component] = False

    return shadow_mask.astype(np.uint8)


def extract_shadow_lengths(
    shadow_mask: np.ndarray,
    building_footprints: Optional[List[Dict]] = None,
    range_direction: str = "horizontal",
) -> List[Dict]:
    """
    Measure shadow extent per building using shadow mask.

    For each shadow region, measures the length in the range direction
    (perpendicular to satellite flight path).

    Args:
        shadow_mask: Binary mask [H, W]
        building_footprints: Optional list of dicts with 'bbox' or 'centroid' keys
                            to associate shadows with buildings
        range_direction: Direction of range ('horizontal' or 'vertical')

    Returns:
        List of dicts with shadow measurements:
            - shadow_length_px: shadow length in pixels
            - centroid: (row, col)
            - area_px: shadow area in pixels
    """
    # Label connected shadow components
    labeled, num_features = ndimage.label(shadow_mask)

    shadow_measurements = []

    for i in range(1, num_features + 1):
        component = labeled == i
        rows, cols = np.where(component)

        if len(rows) == 0:
            continue

        # Measure shadow extent in range direction
        if range_direction == "horizontal":
            shadow_length_px = cols.max() - cols.min() + 1
        else:
            shadow_length_px = rows.max() - rows.min() + 1

        measurement = {
            "shadow_id": i,
            "shadow_length_px": float(shadow_length_px),
            "centroid": (float(rows.mean()), float(cols.mean())),
            "area_px": int(np.sum(component)),
            "bbox": {
                "min_row": int(rows.min()),
                "max_row": int(rows.max()),
                "min_col": int(cols.min()),
                "max_col": int(cols.max()),
            },
        }
        shadow_measurements.append(measurement)

    return shadow_measurements


def shadow_fallback_pipeline(
    sar_image: np.ndarray,
    pixel_spacing_m: float = 0.5,
    incidence_angle_deg: float = DEFAULT_INCIDENCE_ANGLE_DEG,
    threshold_percentile: float = 10.0,
) -> List[Dict]:
    """
    Full shadow-based height estimation pipeline (Plan B).

    Steps:
        1. Detect shadows in SAR image
        2. Measure shadow lengths
        3. Estimate heights using SAR geometry

    Args:
        sar_image: Raw SAR image [C, H, W] or [H, W]
        pixel_spacing_m: Ground pixel spacing in meters
        incidence_angle_deg: Radar incidence angle
        threshold_percentile: Shadow detection threshold

    Returns:
        List of dicts with estimated heights per shadow/building
    """
    # Detect shadows
    shadow_mask = detect_shadows_sar(sar_image, threshold_percentile)

    # Measure shadow lengths
    measurements = extract_shadow_lengths(shadow_mask)

    # Estimate heights
    results = []
    for m in measurements:
        height = estimate_height_from_shadow(
            m["shadow_length_px"],
            pixel_spacing_m,
            incidence_angle_deg,
        )
        result = {
            **m,
            "estimated_height_m": round(height, 2),
            "pixel_spacing_m": pixel_spacing_m,
            "incidence_angle_deg": incidence_angle_deg,
            "method": "SAR_shadow_geometry",
        }
        results.append(result)

    return results
