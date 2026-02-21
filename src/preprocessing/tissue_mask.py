"""Tissue mask utilities for hyperspectral cubes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects


def _validate_cube(cube: np.ndarray) -> None:
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H, W, B), got {cube.shape}")


def _mean_intensity_map(cube: np.ndarray) -> np.ndarray:
    _validate_cube(cube)
    return cube.mean(axis=2, dtype=np.float32)


def build_tissue_mask(
    cube: np.ndarray,
    method: str = "mean_otsu",
    min_object_size: int = 1000,
    min_hole_size: int = 1000,
) -> np.ndarray:
    """
    Build boolean tissue mask from one hyperspectral cube.

    For method='mean_otsu', we threshold the mean intensity image with Otsu and
    pick the darker side as tissue (microscopy background is typically brighter).
    """
    if method != "mean_otsu":
        raise ValueError(f"Unsupported tissue mask method: {method}")

    gray = _mean_intensity_map(cube)
    threshold = threshold_otsu(gray)

    low_mask = gray <= threshold
    high_mask = gray > threshold

    low_mean = float(gray[low_mask].mean()) if low_mask.any() else np.inf
    high_mean = float(gray[high_mask].mean()) if high_mask.any() else np.inf
    mask = low_mask if low_mean <= high_mean else high_mask

    # skimage>=0.26 renamed min_size/area_threshold -> max_size with
    # slightly different boundary semantics (<=). We keep previous behavior
    # by subtracting one from the threshold.
    object_threshold = max(0, min_object_size - 1)
    hole_threshold = max(0, min_hole_size - 1)
    try:
        cleaned = remove_small_objects(mask, max_size=object_threshold)
    except TypeError:
        cleaned = remove_small_objects(mask, min_size=min_object_size)
    try:
        cleaned = remove_small_holes(cleaned, max_size=hole_threshold)
    except TypeError:
        cleaned = remove_small_holes(cleaned, area_threshold=min_hole_size)
    return cleaned.astype(bool)


def tissue_ratio(mask: np.ndarray) -> float:
    """Return tissue fraction in [0, 1]."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    return float(mask.mean())


def save_tissue_mask(mask: np.ndarray, out_path: Path) -> Path:
    """Persist mask as .npy with uint8 values {0,1}."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask.astype(np.uint8))
    return out_path
