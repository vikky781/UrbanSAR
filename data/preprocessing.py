"""
UrbanSAR - SAR & Optical Image Preprocessing

Includes Lee speckle filter for SAR denoising, radiometric normalization,
and optical image normalization.
"""
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d
from typing import Tuple


def lee_speckle_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Lee speckle filter for SAR image denoising.

    The Lee filter reduces multiplicative speckle noise while preserving
    edges and spatial resolution. It estimates the local mean and variance
    to adaptively weight between the observed pixel and the local mean.

    Formula:
        filtered = mean + k * (pixel - mean)
        where k = var_local / (var_local + var_noise)
        and var_noise is estimated from local statistics

    Args:
        image: SAR image array, shape [C, H, W] or [H, W]
        window_size: Size of the local window (default: 7)

    Returns:
        Filtered image, same shape as input
    """
    if image.ndim == 3:
        # Apply filter to each channel independently
        filtered = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            filtered[c] = _lee_filter_2d(image[c], window_size)
        return filtered
    else:
        return _lee_filter_2d(image, window_size)


def _lee_filter_2d(image: np.ndarray, window_size: int) -> np.ndarray:
    """Apply Lee filter to a single 2D image."""
    img = image.astype(np.float64)

    # Compute local mean and variance
    local_mean = uniform_filter(img, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(img ** 2, size=window_size, mode="reflect")
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Ensure non-negative

    # Estimate noise variance (using overall image statistics)
    # For multiplicative noise model: noise_var ≈ mean(local_var) / mean(local_mean)^2
    overall_mean = np.mean(local_mean)
    if overall_mean > 0:
        noise_var = np.mean(local_var) / (overall_mean ** 2 + 1e-10)
    else:
        noise_var = 1e-10

    # Compute weighting factor k
    # k = 1 means keep original (high local variance = edge)
    # k = 0 means use local mean (low local variance = homogeneous)
    denominator = local_var + noise_var * (local_mean ** 2)
    k = local_var / (denominator + 1e-10)
    k = np.clip(k, 0, 1)

    # Apply filter
    filtered = local_mean + k * (img - local_mean)
    return filtered.astype(np.float32)


def radiometric_normalize(sar_image: np.ndarray) -> np.ndarray:
    """
    Radiometric normalization for SAR imagery.

    Converts linear intensity to dB scale, then normalizes to [0, 1].
    dB = 10 * log10(intensity)

    Args:
        sar_image: SAR image in linear intensity, shape [C, H, W] or [H, W]

    Returns:
        Normalized image in [0, 1] range
    """
    # Avoid log of zero
    img = np.maximum(sar_image, 1e-10).astype(np.float64)

    # Convert to dB
    img_db = 10.0 * np.log10(img)

    # Clip extreme values (typical SAR range in dB)
    # Using percentiles for robustness against outliers
    if img_db.ndim == 3:
        for c in range(img_db.shape[0]):
            p2, p98 = np.percentile(img_db[c], [2, 98])
            img_db[c] = np.clip(img_db[c], p2, p98)
            # Min-max normalize per channel
            range_val = p98 - p2
            if range_val > 0:
                img_db[c] = (img_db[c] - p2) / range_val
            else:
                img_db[c] = 0.0
    else:
        p2, p98 = np.percentile(img_db, [2, 98])
        img_db = np.clip(img_db, p2, p98)
        range_val = p98 - p2
        if range_val > 0:
            img_db = (img_db - p2) / range_val
        else:
            img_db = 0.0

    return img_db.astype(np.float32)


def normalize_optical(image: np.ndarray) -> np.ndarray:
    """
    Normalize optical satellite image to [0, 1] range.

    Uses per-channel min-max normalization with percentile clipping
    for robustness against outliers and sensor artifacts.

    Args:
        image: Optical image, shape [C, H, W] or [H, W].
               Typically uint8 [0-255] or uint16 [0-65535].

    Returns:
        Normalized image in [0, 1] range
    """
    img = image.astype(np.float32)

    if img.ndim == 3:
        for c in range(img.shape[0]):
            p2, p98 = np.percentile(img[c], [2, 98])
            img[c] = np.clip(img[c], p2, p98)
            range_val = p98 - p2
            if range_val > 0:
                img[c] = (img[c] - p2) / range_val
            else:
                img[c] = 0.0
    else:
        p2, p98 = np.percentile(img, [2, 98])
        img = np.clip(img, p2, p98)
        range_val = p98 - p2
        if range_val > 0:
            img = (img - p2) / range_val
        else:
            img = 0.0

    return img


def preprocess_pair(
    sar: np.ndarray,
    optical: np.ndarray,
    apply_lee: bool = True,
    lee_window: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a SAR-optical pair.

    SAR pipeline:    Lee speckle filter → radiometric normalization
    Optical pipeline: percentile normalization

    Args:
        sar: Raw SAR image, shape [C, H, W]
        optical: Raw optical image, shape [C, H, W]
        apply_lee: Whether to apply Lee speckle filter
        lee_window: Lee filter window size

    Returns:
        Tuple of (preprocessed_sar, preprocessed_optical), both [0, 1]
    """
    # SAR preprocessing
    sar_processed = sar.copy()
    if apply_lee:
        sar_processed = lee_speckle_filter(sar_processed, window_size=lee_window)
    sar_processed = radiometric_normalize(sar_processed)

    # Optical preprocessing
    optical_processed = normalize_optical(optical.copy())

    return sar_processed, optical_processed
