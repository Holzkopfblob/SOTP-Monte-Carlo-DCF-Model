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


def conditional_sensitivity(
    equity_values: np.ndarray,
    input_samples: dict[str, np.ndarray],
    *,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> dict[str, dict[str, float]]:
    """Spearman sensitivity split by outcome regime.

    Returns ``{"bear": {...}, "bull": {...}}`` where bear = bottom quantile
    and bull = top quantile of equity values.
    """
    q_lo = float(np.quantile(equity_values, lower_quantile))
    q_hi = float(np.quantile(equity_values, upper_quantile))
    mask_bear = equity_values <= q_lo
    mask_bull = equity_values >= q_hi

    bear: dict[str, float] = {}
    bull: dict[str, float] = {}
    for name, samples in input_samples.items():
        if np.std(samples) < 1e-12:
            continue
        # Bear case
        if np.sum(mask_bear) > 10 and np.std(samples[mask_bear]) > 1e-12:
            corr_b, _ = sp_stats.spearmanr(samples[mask_bear], equity_values[mask_bear])
            if np.isfinite(corr_b):
                bear[name] = float(corr_b)
        # Bull case
        if np.sum(mask_bull) > 10 and np.std(samples[mask_bull]) > 1e-12:
            corr_u, _ = sp_stats.spearmanr(samples[mask_bull], equity_values[mask_bull])
            if np.isfinite(corr_u):
                bull[name] = float(corr_u)

    return {
        "bear": dict(sorted(bear.items(), key=lambda x: abs(x[1]), reverse=True)),
        "bull": dict(sorted(bull.items(), key=lambda x: abs(x[1]), reverse=True)),
    }


def compute_statistics(arr: np.ndarray) -> dict[str, float]:
    """Full descriptive statistics for a 1-D array of outcomes."""
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "Mittelwert":  mean,
        "Median":      float(np.median(arr)),
        "Std.-Abw.":   std,
        "Schiefe":     float(sp_stats.skew(arr)),
        "Kurtosis":    float(sp_stats.kurtosis(arr)),
        "CV":          abs(std / mean) if abs(mean) > 1e-12 else 0.0,
        "IQR":         float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "P5 (5%)":     float(np.percentile(arr, 5)),
        "P25 (25%)":   float(np.percentile(arr, 25)),
        "P75 (75%)":   float(np.percentile(arr, 75)),
        "P95 (95%)":   float(np.percentile(arr, 95)),
        "Min":         float(np.min(arr)),
        "Max":         float(np.max(arr)),
    }


def compute_tail_risk(
    arr: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Value-at-Risk, Conditional VaR (Expected Shortfall), and Tail Ratio.

    Parameters
    ----------
    arr : (n,) array
        Distribution of values (e.g. equity values or returns).
    alpha : float
        Tail quantile (default 5 %).

    Returns
    -------
    dict with ``var``, ``cvar``, ``tail_ratio``.
    """
    var = float(np.percentile(arr, alpha * 100))
    tail_mask = arr <= var
    cvar = float(np.mean(arr[tail_mask])) if np.any(tail_mask) else var
    # Tail ratio = |P95 / P5| – measures asymmetry of tails
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    tail_ratio = abs(p95 / p5) if abs(p5) > 1e-12 else 0.0
    return {"var": var, "cvar": cvar, "tail_ratio": tail_ratio}


def normality_test(arr: np.ndarray) -> dict[str, float]:
    """Jarque-Bera and Shapiro-Wilk normality tests.

    Returns p-values and test statistics.  A low p-value (< 0.05) indicates
    the distribution is NOT normal.

    For arrays > 5000 elements, Shapiro-Wilk uses a subsample (limit of the
    scipy implementation).
    """
    # Jarque-Bera (works for any size)
    jb_stat, jb_p = sp_stats.jarque_bera(arr)

    # Shapiro-Wilk (requires n ≤ 5000)
    n = len(arr)
    if n > 5000:
        subsample = np.random.default_rng(42).choice(arr, 5000, replace=False)
    else:
        subsample = arr
    sw_stat, sw_p = sp_stats.shapiro(subsample)

    skewness = float(sp_stats.skew(arr))
    kurt = float(sp_stats.kurtosis(arr))

    # Recommendation
    is_normal = jb_p > 0.05 and sw_p > 0.05
    if is_normal:
        recommendation = "Normal"
    elif skewness > 0.5:
        recommendation = "Lognormal"
    elif skewness < -0.5:
        recommendation = "Lognormal (gespiegelt)"
    else:
        recommendation = "Skew-Normal"

    return {
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "shapiro_wilk_stat": float(sw_stat),
        "shapiro_wilk_p": float(sw_p),
        "skewness": skewness,
        "kurtosis": kurt,
        "is_normal": float(is_normal),
        "recommendation": recommendation,
    }
