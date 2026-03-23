"""
Shared constants and helpers for portfolio page modules.
"""
from __future__ import annotations

from collections import OrderedDict

from application.portfolio_service import PortfolioResult

# Fair-value distribution options for the portfolio app.
# NOTE: These are portfolio-specific (include DCF-App bridge, exclude "Fest").
PORTFOLIO_DIST_OPTIONS = [
    "Aus DCF-App (μ, σ, Schiefe)",
    "Normal",
    "Lognormal",
    "PERT",
    "Dreiecksverteilung",
    "Gleichverteilung",
]

SECTOR_LIST = [
    "Technologie", "Gesundheit", "Finanzen", "Industrie",
    "Konsumgüter", "Energie", "Immobilien", "Versorger",
    "Telekommunikation", "Grundstoffe", "Sonstige",
]

METHOD_ORDER = [
    "Gleichgewicht (1/N)",
    "Max Sharpe",
    "Min Volatilität",
    "Risk Parity",
    "Min CVaR",
    "Max Diversifikation",
    "Kelly (Multi-Asset)",
    "HRP",
    "Black-Litterman",
]

METHOD_SYMBOLS = {
    "Gleichgewicht (1/N)": "circle",
    "Max Sharpe": "star",
    "Min Volatilität": "diamond",
    "Risk Parity": "square",
    "Min CVaR": "hexagon",
    "Max Diversifikation": "cross",
    "Kelly (Multi-Asset)": "pentagon",
    "HRP": "triangle-up",
    "Black-Litterman": "bowtie",
}

METHOD_COLORS = {
    "Gleichgewicht (1/N)": "#7f7f7f",
    "Max Sharpe": "#1f77b4",
    "Min Volatilität": "#17becf",
    "Risk Parity": "#ff7f0e",
    "Min CVaR": "#d62728",
    "Max Diversifikation": "#2ca02c",
    "Kelly (Multi-Asset)": "#9467bd",
    "HRP": "#e377c2",
    "Black-Litterman": "#bcbd22",
}


def active_results(pf: dict) -> OrderedDict[str, PortfolioResult]:
    """Return non-None optimisation results in the canonical display order."""
    out: OrderedDict[str, PortfolioResult] = OrderedDict()
    for key in METHOD_ORDER:
        res = pf["opt_results"].get(key)
        if res is not None:
            out[key] = res
    return out
