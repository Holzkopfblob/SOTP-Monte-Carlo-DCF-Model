"""
Descriptive Statistics Utilities.

Pure functions operating on numpy arrays – no I/O, no side-effects.
Extracted from ``application.simulation_service`` so that any layer can
reuse them without importing the full service.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def compute_sensitivity(
    equity_values: np.ndarray,
    input_samples: dict[str, np.ndarray],
) -> dict[str, float]:
    """Spearman rank-correlation of each stochastic input with equity value.

    Returns a dict sorted by **absolute** correlation (descending).
    Constant inputs (σ ≈ 0) are silently excluded.

    This is a pure domain function – no dependency on the application layer.
    """
    correlations: dict[str, float] = {}
    for name, samples in input_samples.items():
        if np.std(samples) < 1e-12:
            continue
        corr, _ = sp_stats.spearmanr(samples, equity_values)
        if np.isfinite(corr):
            correlations[name] = float(corr)
    return dict(
        sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    )

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
