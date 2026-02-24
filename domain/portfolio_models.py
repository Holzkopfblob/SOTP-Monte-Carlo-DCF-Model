"""
Domain Models for the Portfolio Optimisation Module.

Pure value objects / data-transfer structures – no I/O, no side-effects.
Previously housed inside ``application.portfolio_service``; moved here
so that every layer can import them without pulling in the service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class CovarianceMethod(str, Enum):
    """Covariance estimation method for portfolio optimisation."""
    SAMPLE   = "Sample-Kovarianz (Standard)"
    LEDOIT_WOLF = "Ledoit-Wolf Shrinkage"


# ═══════════════════════════════════════════════════════════════════════════
# Black-Litterman Views
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InvestorView:
    """One absolute or relative investor view for Black-Litterman.

    *absolute* view: "Asset i will return q %"
    *relative* view: "Asset i will outperform j by q %" (not yet supported)
    """
    asset_index: int
    expected_return: float   # absolute expected return (decimal)
    confidence: float = 0.5  # 0..1  (higher = more confident)


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


# ═══════════════════════════════════════════════════════════════════════════
# Historical crisis scenarios
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HistoricalScenario:
    """Predefined historical crisis scenario with sector-specific shocks."""
    name: str
    description: str
    market_shock_pct: float              # uniform market drawdown %
    corr_stress: float                   # minimum off-diagonal correlation
    sector_shocks: dict[str, float]      # sector → additional shock %
    duration_months: int = 12            # typical drawdown duration


HISTORICAL_SCENARIOS: dict[str, HistoricalScenario] = {
    "COVID-19 Crash (2020)": HistoricalScenario(
        name="COVID-19 Crash (2020)",
        description="Pandemie-Schock: schneller Abverkauf, V-förmige Erholung. "
                    "Tourismus, Energie und Industrie am stärksten betroffen.",
        market_shock_pct=-35.0,
        corr_stress=0.90,
        sector_shocks={
            "Energie": -25.0, "Industrie": -15.0, "Konsumgüter": -10.0,
            "Immobilien": -10.0,
        },
        duration_months=3,
    ),
    "GFC / Finanzkrise (2008)": HistoricalScenario(
        name="GFC / Finanzkrise (2008)",
        description="Globale Bankenkrise: systemisches Risiko, Kreditklemme. "
                    "Finanzsektor und Immobilien kollabieren.",
        market_shock_pct=-50.0,
        corr_stress=0.95,
        sector_shocks={
            "Finanzen": -40.0, "Immobilien": -35.0, "Industrie": -20.0,
            "Konsumgüter": -15.0,
        },
        duration_months=18,
    ),
    "Dot-Com Crash (2001)": HistoricalScenario(
        name="Dot-Com Crash (2001)",
        description="Technologieblase platzt: massive Überbewertungen im Tech-Sektor "
                    "korrigieren sich. Defensive Sektoren relativ stabil.",
        market_shock_pct=-40.0,
        corr_stress=0.75,
        sector_shocks={
            "Technologie": -55.0, "Telekommunikation": -30.0,
        },
        duration_months=30,
    ),
    "Euro-Krise (2011)": HistoricalScenario(
        name="Euro-Krise (2011)",
        description="Europäische Schuldenkrise: Staatsanleihen-Spread-Ausweitung, "
                    "Banken unter Druck, Austeritätspolitik.",
        market_shock_pct=-25.0,
        corr_stress=0.85,
        sector_shocks={
            "Finanzen": -30.0, "Immobilien": -15.0,
        },
        duration_months=9,
    ),
    "Inflation Shock (2022)": HistoricalScenario(
        name="Inflation Shock (2022)",
        description="Zinswende nach Jahrzehnt der Niedrigzinsen: Duration-Assets "
                    "und Growth-Titel verlieren, Value-Titel outperformen.",
        market_shock_pct=-20.0,
        corr_stress=0.70,
        sector_shocks={
            "Technologie": -30.0, "Immobilien": -20.0,
            "Energie": +15.0, "Grundstoffe": +5.0,
        },
        duration_months=10,
    ),
    "Milde Korrektur": HistoricalScenario(
        name="Milde Korrektur",
        description="Typische 10–15 %-Korrektur ohne systemisches Ereignis.",
        market_shock_pct=-15.0,
        corr_stress=0.70,
        sector_shocks={},
        duration_months=4,
    ),
}


# Macro factor sensitivity by sector (% return change per +1% factor move)
MACRO_SECTOR_SENSITIVITY: dict[str, dict[str, float]] = {
    "Technologie":       {"Zinsen": -2.5, "Inflation": -1.5, "BIP": +1.8},
    "Konsumgüter":       {"Zinsen": -1.0, "Inflation": -1.2, "BIP": +1.5},
    "Industrie":         {"Zinsen": -1.2, "Inflation": -0.8, "BIP": +2.0},
    "Grundstoffe":       {"Zinsen": -0.8, "Inflation": +0.5, "BIP": +1.8},
    "Gesundheit":        {"Zinsen": -0.5, "Inflation": -0.3, "BIP": +0.5},
    "Versorger":         {"Zinsen": -2.0, "Inflation": -0.5, "BIP": +0.3},
    "Telekommunikation": {"Zinsen": -1.5, "Inflation": -0.5, "BIP": +0.5},
    "Finanzen":          {"Zinsen": +1.5, "Inflation": -0.5, "BIP": +1.5},
    "Immobilien":        {"Zinsen": -3.0, "Inflation": -0.8, "BIP": +1.0},
    "Energie":           {"Zinsen": -0.5, "Inflation": +1.0, "BIP": +1.5},
    "Sonstige":          {"Zinsen": -1.0, "Inflation": -0.5, "BIP": +1.0},
}
