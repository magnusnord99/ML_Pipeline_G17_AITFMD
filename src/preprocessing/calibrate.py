"""Calibration utilities for HistologyHSI-GB cubes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import spectral


def load_envi_cube(hdr_path: Path, bin_path: Path) -> np.ndarray:
    """Load ENVI cube and return float32 array with shape (H, W, B)."""
    img = spectral.envi.open(str(hdr_path), str(bin_path))
    cube = np.asarray(img.load(), dtype=np.float32)
    return cube


def calibrate_cube(
    raw: np.ndarray,
    dark: np.ndarray,
    white: np.ndarray,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Apply flat-field calibration and clipping."""
    calibrated = (raw - dark) / (white - dark + eps)
    calibrated = np.clip(calibrated, clip_min, clip_max)
    return calibrated.astype(np.float32)


def reduce_bands_neighbor_average(cube: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Reduce spectral bands by averaging adjacent neighbors.

    Example: B=826 and window=3 -> floor(826/3)=275 bands.
    """
    if window <= 1:
        return cube

    h, w, b = cube.shape
    b_reduced = b // window
    if b_reduced == 0:
        raise ValueError(f"window={window} is too large for b={b}")

    trimmed = cube[:, :, : b_reduced * window]
    reshaped = trimmed.reshape(h, w, b_reduced, window)
    reduced = reshaped.mean(axis=3)
    return reduced.astype(np.float32)


def output_npy_path(calibrated_dir: Path, patient_id: str, roi_name: str) -> Path:
    """Return output path for one calibrated ROI cube."""
    return calibrated_dir / patient_id / f"{roi_name}.npy"

