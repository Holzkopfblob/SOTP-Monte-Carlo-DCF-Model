"""Shared metric builders/renderers for portfolio result pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.portfolio_service import AssetMetrics, PortfolioResult


def build_single_summary_rows(asset_metrics: list[AssetMetrics]) -> list[dict]:
    """Build summary rows for single-asset table."""
    summary_rows = []
    for am in asset_metrics:
        summary_rows.append({
            "Asset": am.name,
            "Sektor": am.sector,
            "Kurs": f"{am.current_price:,.2f} €",
            "Ø Fair Value": f"{am.mean_fv:,.2f} €",
            "E[Rendite]": f"{am.expected_return:+.1%}",
            "P(Gewinn)": f"{am.prob_profit:.1%}",
            "MoS": f"{am.margin_of_safety:+.1%}",
            "Kelly f*": f"{am.kelly_fraction:.1%}",
            "VaR(5%)": f"{am.var_5:+.1%}",
            "CVaR(5%)": f"{am.cvar_5:+.1%}",
            "Sortino": f"{am.sortino_ratio:.2f}",
            "Omega": f"{am.omega_ratio:.2f}",
            "Signal": am.signal,
        })
    return summary_rows


def render_single_asset_metric_grids(am: AssetMetrics) -> None:
    """Render the two metric rows for an asset detail tab."""
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("E[Rendite]", f"{am.expected_return:+.1%}")
    m2.metric("P(Gewinn)", f"{am.prob_profit:.1%}")
    m3.metric("MoS", f"{am.margin_of_safety:+.1%}")
    m4.metric("Half Kelly", f"{am.half_kelly:.1%}")
    m5.metric("Sortino", f"{am.sortino_ratio:.2f}")
    m6.metric("Omega", f"{am.omega_ratio:.2f}")

    m7, m8, m9, m10, m11, _ = st.columns(6)
    m7.metric("FV (P5)", f"{am.fv_p5:,.2f} €")
    m8.metric("FV (P50)", f"{am.median_fv:,.2f} €")
    m9.metric("FV (P95)", f"{am.fv_p95:,.2f} €")
    m10.metric("VaR (5%)", f"{am.var_5:+.1%}")
    m11.metric("CVaR (5%)", f"{am.cvar_5:+.1%}")


def build_portfolio_weights_table(
    names: list[str],
    active: dict[str, PortfolioResult],
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Build formatted table and raw weight dict for portfolio methods."""
    weights_dict: dict[str, list[float]] = {}
    table_data: dict[str, list] = {"Asset": names}
    for method_name, pr in active.items():
        table_data[method_name] = [f"{w:.1%}" for w in pr.weights]
        weights_dict[method_name] = pr.weights
    return pd.DataFrame(table_data), weights_dict


def build_portfolio_metrics_rows(active: dict[str, PortfolioResult]) -> list[dict]:
    """Build portfolio-level metric rows per optimization method."""
    metrics_rows = []
    for pr in active.values():
        metrics_rows.append({
            "Methode": pr.name,
            "E[Rendite]": f"{pr.expected_return:+.1%}",
            "Volatilität": f"{pr.volatility:.1%}",
            "Sharpe Ratio": f"{pr.sharpe_ratio:.2f}",
            "VaR (5%)": f"{pr.var_5:+.1%}",
            "CVaR (5%)": f"{pr.cvar_5:+.1%}",
            "P(Verlust)": f"{pr.prob_loss:.1%}",
            "Div.-Ratio": f"{pr.diversification_ratio:.2f}",
            "Eff. # Assets": f"{pr.effective_n_assets:.1f}",
        })
    return metrics_rows
