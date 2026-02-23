"""
Descriptive Statistics Utilities.

Pure functions operating on numpy arrays – no I/O, no side-effects.
Extracted from ``application.simulation_service`` so that any layer can
reuse them without importing the full service.
"""
from __future__ import annotations

import numpy as np


def compute_statistics(arr: np.ndarray) -> dict[str, float]:
    """Full descriptive statistics for a 1-D array of outcomes."""
    return {
        "Mittelwert":  float(np.mean(arr)),
        "Median":      float(np.median(arr)),
        "Std.-Abw.":   float(np.std(arr)),
        "P5 (5%)":     float(np.percentile(arr, 5)),
        "P25 (25%)":   float(np.percentile(arr, 25)),
        "P75 (75%)":   float(np.percentile(arr, 75)),
        "P95 (95%)":   float(np.percentile(arr, 95)),
        "Min":         float(np.min(arr)),
        "Max":         float(np.max(arr)),
    }
