"""
Domain Models for the Portfolio Optimisation Module.

Pure value objects / data-transfer structures – no I/O, no side-effects.
Previously housed inside ``application.portfolio_service``; moved here
so that every layer can import them without pulling in the service.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Input
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetInput:
    """One asset to include in the portfolio analysis."""
    name: str
    sector: str
    current_price: float
    fv_samples: np.ndarray          # (n_sim,) MC fair-value samples
    min_weight: float = 0.0
    max_weight: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Single-asset analysis result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetMetrics:
    """Comprehensive single-asset analysis results."""
    name: str
    sector: str
    current_price: float

    # Fair-value statistics
    mean_fv: float
    median_fv: float
    fv_std: float
    fv_p5: float
    fv_p25: float
    fv_p75: float
    fv_p95: float

    # Return metrics
    expected_return: float
    return_std: float
    prob_profit: float
    margin_of_safety: float

    # Kelly
    kelly_fraction: float
    half_kelly: float

    # Scenario percentiles (as returns)
    upside_p75: float
    downside_p25: float

    # Risk metrics
    var_5: float
    cvar_5: float
    sortino_ratio: float
    omega_ratio: float

    # Composite signal
    signal: str          # "🟢 Kaufen" | "🟡 Halten" | "🔴 Meiden"

    # Raw vectors for downstream use (excluded from repr)
    returns: np.ndarray = field(repr=False)
    fv_samples: np.ndarray = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio-level results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    """Result for one optimisation method."""
    name: str
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_5: float
    cvar_5: float
    prob_loss: float
    diversification_ratio: float
    effective_n_assets: float


@dataclass
class StressTestResult:
    """Stress test output for one portfolio method."""
    method_name: str
    return_normal: float
    return_stressed: float
    delta_return: float
    vol_stressed: float
    var_5_stressed: float
    cvar_5_stressed: float
    prob_loss: float
