"""
Portfolio Efficient Frontier Tab
==================================
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from presentation.charts import (
    COLORS,
    PALETTE_EXTENDED,
    TEMPLATE,
    correlation_heatmap,
)
from presentation.pages.pf_common import active_results


def render_frontier(tab) -> None:
    """Render Tab 4 (Efficient Frontier)."""
    with tab:
        if st.session_state.pf_results is None:
            st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
            return
        if len(st.session_state.pf_results["asset_metrics"]) < 2:
            st.info("Die Efficient Frontier benötigt mindestens 2 Assets.")
            return

        pf = st.session_state.pf_results
        active = active_results(pf)
        st.header("📈 Efficient Frontier")

        with st.expander("ℹ️ Was ist die Efficient Frontier?", expanded=False):
            st.markdown(r"""
Die Efficient Frontier zeigt alle Portfolios mit **maximaler Rendite
für ein gegebenes Risiko** (oder minimalem Risiko für gegebene Rendite).

**Capital Market Line (CML):**
$E[R_P] = r_f + \frac{E[R_M] - r_f}{\sigma_M} \cdot \sigma_P$

Portfolios *auf* der Frontier sind **effizient** – kein anderes Portfolio
bietet bei gleichem Risiko mehr Rendite.
""")

        names = [am.name for am in pf["asset_metrics"]]
        fig_ef = go.Figure()

        # Frontier curve
        if len(pf["ef_vols"]) > 0:
            fig_ef.add_trace(go.Scatter(
                x=pf["ef_vols"] * 100, y=pf["ef_rets"] * 100,
                mode="lines", name="Efficient Frontier",
                line=dict(color=COLORS["primary"], width=3),
            ))

        # Individual assets
        for idx, am in enumerate(pf["asset_metrics"]):
            fig_ef.add_trace(go.Scatter(
                x=[am.return_std * 100],
                y=[am.expected_return * 100],
                mode="markers+text",
                name=am.name,
                marker=dict(
                    size=12,
                    color=PALETTE_EXTENDED[idx % len(PALETTE_EXTENDED)],
                ),
                text=[am.name],
                textposition="top center",
            ))

        # Optimised portfolio points
        symbols = {
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
        point_colors = {
            "Gleichgewicht (1/N)": COLORS["neutral"],
            "Max Sharpe": COLORS["primary"],
            "Min Volatilität": "#17becf",
            "Risk Parity": COLORS["secondary"],
            "Min CVaR": COLORS["negative"],
            "Max Diversifikation": COLORS["positive"],
            "Kelly (Multi-Asset)": COLORS["accent"],
            "HRP": "#e377c2",
            "Black-Litterman": "#bcbd22",
        }

        for method_name, pr in active.items():
            ret = pr.expected_return * 100
            vol = pr.volatility * 100
            fig_ef.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                name=method_name,
                marker=dict(
                    size=16,
                    color=point_colors.get(method_name, COLORS["accent"]),
                    symbol=symbols.get(method_name, "circle"),
                    line=dict(width=2, color="white"),
                ),
                text=[method_name],
                textposition="bottom center",
                textfont=dict(size=9),
            ))

        # Capital Market Line
        ms = active.get("Max Sharpe")
        if ms is not None and ms.volatility > 1e-12:
            slope = (ms.expected_return - pf["rf"]) / ms.volatility
            x_max = max(pf["std_vec"].max() * 1.2, ms.volatility * 1.5)
            x_cml = np.linspace(0, x_max, 50)
            y_cml = pf["rf"] + slope * x_cml
            fig_ef.add_trace(go.Scatter(
                x=x_cml * 100, y=y_cml * 100,
                mode="lines", name="Capital Market Line",
                line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
            ))

        fig_ef.update_layout(
            title="Efficient Frontier & Portfolio-Positionen",
            xaxis_title="Volatilität (%)",
            yaxis_title="Erwartete Rendite (%)",
            template=TEMPLATE, height=600,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # ── Correlation heatmap ───────────────────────────────────────
        st.divider()
        st.subheader("🔥 Korrelationsmatrix (Heatmap)")
        st.plotly_chart(
            correlation_heatmap(pf["corr_matrix"], names),
            use_container_width=True,
        )
