"""Historical scenario section for portfolio stress page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.portfolio_stress import PortfolioStressTester
from domain.portfolio_models import HISTORICAL_SCENARIOS


def render_stress_historical_section(pf: dict, active: dict) -> None:
    """Render historical stress scenario comparison block."""
    st.divider()
    st.subheader("📜 Historische Krisenszenarien")

    with st.expander("ℹ️ Was wird hier verglichen?", expanded=False):
        st.markdown("""
Jedes historische Szenario wendet **sektorspezifische Schocks** auf
Ihr Portfolio an, basierend auf realen Krisen.  Korrelationen werden
analog zur tatsächlichen Marktdynamik angehoben.

So sehen Sie auf einen Blick, wie sich Ihr Portfolio in vergangenen
Krisen verhalten hätte — und welche Methode die robusteste ist.
""")

    if st.button("📜 Alle historischen Szenarien berechnen", use_container_width=True):
        tester = PortfolioStressTester(risk_free_rate=pf["rf"])
        portfolio_weights = {name: pr.weights for name, pr in active.items()}

        scenario_rows = []
        for sc_name, scenario in HISTORICAL_SCENARIOS.items():
            sc_results, _ = tester.stress_test_scenario(
                scenario, portfolio_weights,
                pf["returns_matrix"], pf["sectors"],
            )
            for sr in sc_results:
                scenario_rows.append({
                    "Szenario": sc_name,
                    "Methode": sr.method_name,
                    "Δ Rendite": f"{sr.delta_return:+.1%}",
                    "CVaR 5%": f"{sr.cvar_5_stressed:+.1%}",
                    "P(Verlust)": f"{sr.prob_loss:.1%}",
                })

        if scenario_rows:
            st.dataframe(
                pd.DataFrame(scenario_rows),
                use_container_width=True, hide_index=True,
            )

            for sc_name, scenario in HISTORICAL_SCENARIOS.items():
                with st.expander(f"📋 {sc_name}", expanded=False):
                    st.markdown(f"""
**{scenario.description}**

| Parameter | Wert |
|---|---|
| Marktschock | {scenario.market_shock_pct:+.0f} % |
| Korrelation ≥ | {scenario.corr_stress:.2f} |
| Dauer | ~{scenario.duration_months} Monate |
| Sektorschocks | {', '.join(f'{s}: {v:+.0f}%' for s, v in scenario.sector_shocks.items()) or 'Keine'} |
""")
