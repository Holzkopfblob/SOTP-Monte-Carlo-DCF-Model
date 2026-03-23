"""Macro factor sensitivity section for portfolio stress page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.portfolio_stress import PortfolioStressTester


def render_stress_macro_section(pf: dict, active: dict) -> None:
    """Render macro factor sensitivity controls and impact tables."""
    st.divider()
    st.subheader("🌍 Makrofaktor-Sensitivität")

    with st.expander("ℹ️ Hinweis zur Makro-Analyse", expanded=False):
        st.markdown("""
Diese Analyse schätzt, wie sich **Zinsänderungen**, **Inflationserwartungen**
und **BIP-Wachstumsänderungen** auf Ihr Portfolio auswirken könnten.

Die Sensitivitäten basieren auf empirischen Sektorstudien und sind
**Näherungswerte**, keine Prognosen. Die Analyse hilft bei der
Einordnung der Portfolioexposition gegenüber Makrofaktoren.
""")

    mc1, mc2, mc3 = st.columns(3)
    ir_delta = mc1.slider(
        "Δ Langfristzins (pp)", min_value=-3.0, max_value=3.0,
        value=0.0, step=0.25, key="macro_ir",
    )
    infl_delta = mc2.slider(
        "Δ Inflation (pp)", min_value=-3.0, max_value=3.0,
        value=0.0, step=0.25, key="macro_infl",
    )
    gdp_delta = mc3.slider(
        "Δ BIP-Wachstum (pp)", min_value=-3.0, max_value=3.0,
        value=0.0, step=0.25, key="macro_gdp",
    )

    if ir_delta != 0 or infl_delta != 0 or gdp_delta != 0:
        impacts = PortfolioStressTester.macro_factor_impact(
            pf["sectors"], ir_delta, infl_delta, gdp_delta,
        )

        impact_rows = []
        for i, am in enumerate(pf["asset_metrics"]):
            impact_rows.append({
                "Asset": am.name,
                "Sektor": am.sector,
                "Impact (pp)": f"{impacts[i]*100:+.2f}",
            })
        st.dataframe(
            pd.DataFrame(impact_rows),
            use_container_width=True, hide_index=True,
        )

        st.markdown("**Portfolio-Impact je Methode:**")
        port_impact_rows = []
        for method_name, pr in active.items():
            port_impact = float(pr.weights @ impacts)
            port_impact_rows.append({
                "Methode": method_name,
                "Portfolio-Impact (pp)": f"{port_impact*100:+.2f}",
            })
        st.dataframe(
            pd.DataFrame(port_impact_rows),
            use_container_width=True, hide_index=True,
        )
