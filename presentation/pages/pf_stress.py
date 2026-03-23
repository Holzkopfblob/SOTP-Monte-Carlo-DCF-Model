"""
Portfolio Stress-Test Tab
=========================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.portfolio_service import PortfolioAnalyser
from presentation.charts import stress_comparison_chart
from presentation.pages.pf_common import active_results
from presentation.layout.states import render_warning_state
from presentation.pages.pf_sections import (
    render_stress_historical_section,
    render_stress_macro_section,
    render_stress_preset_section,
)


def render_stress(tab) -> None:
    """Render Tab 5 (Stress-Tests & Szenario-Analyse)."""
    with tab:
        if st.session_state.pf_results is None:
            render_warning_state("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
            return

        pf = st.session_state.pf_results
        active = active_results(pf)
        st.header("⚡ Stress-Tests & Szenario-Analyse")

        with st.expander("ℹ️ Warum Stress-Tests?", expanded=False):
            st.markdown("""
Stress-Tests prüfen die **Robustheit** Ihres Portfolios.

| Szenario | Beispiel |
|---|---|
| Marktcrash −30 % | COVID-19 (Feb–Mar 2020) |
| Sektorkrise −50 % | Bankenaktien 2008/09 |
| Korrelationsanstieg → 0.9 | Lehman-Krise 2008 |

> **Ziel:** Kein Portfolio sollte unter realistischen Stress-Szenarien
> zu einem untragbaren Verlust führen.
""")

        market_shock, corr_stress, shock_sector, sector_shock_pct = render_stress_preset_section(pf)

        if st.button("⚡ Stress-Test durchführen", type="primary",
                      use_container_width=True):

            analyser = PortfolioAnalyser(risk_free_rate=pf["rf"])

            portfolio_weights = {
                name: pr.weights for name, pr in active.items()
            }

            stress_results, stressed_returns = analyser.stress_test(
                portfolios=portfolio_weights,
                returns_matrix=pf["returns_matrix"],
                asset_sectors=pf["sectors"],
                market_shock_pct=float(market_shock),
                corr_stress=float(corr_stress),
                sector_shock=(
                    shock_sector if shock_sector != "Keiner" else None
                ),
                sector_shock_pct=float(sector_shock_pct),
            )

            # ── Results table ─────────────────────────────────────────
            st.subheader("📊 Stress-Test-Ergebnisse")

            stress_rows = []
            for sr in stress_results:
                stress_rows.append({
                    "Methode": sr.method_name,
                    "E[R] Normal": f"{sr.return_normal:+.1%}",
                    "E[R] Stress": f"{sr.return_stressed:+.1%}",
                    "Δ Rendite": f"{sr.delta_return:+.1%}",
                    "Vol (Stress)": f"{sr.vol_stressed:.1%}",
                    "VaR 5%": f"{sr.var_5_stressed:+.1%}",
                    "CVaR 5%": f"{sr.cvar_5_stressed:+.1%}",
                    "P(Verlust)": f"{sr.prob_loss:.1%}",
                })

            st.dataframe(
                pd.DataFrame(stress_rows),
                use_container_width=True, hide_index=True,
            )

            # ── Stress chart ──────────────────────────────────────────
            st.divider()
            st.subheader("📈 Normal vs. Stress-Szenario")

            sel_stress_method = st.selectbox(
                "Methode für Detail-Ansicht",
                [sr.method_name for sr in stress_results],
                key="stress_method_sel",
            )

            sel_w = active[sel_stress_method].weights

            normal_port = pf["returns_matrix"] @ sel_w
            stressed_port = stressed_returns @ sel_w

            enable_heavy_stress_chart = st.checkbox(
                "Detailchart laden",
                value=False,
                key="stress_lazy_chart",
                help="Lädt die überlagerte Histogramm-Visualisierung nur bei Bedarf.",
            )
            if enable_heavy_stress_chart:
                st.plotly_chart(
                    stress_comparison_chart(
                        normal_port, stressed_port, sel_stress_method,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Detailchart ist deaktiviert. Aktivieren Sie 'Detailchart laden' für die Visualisierung.")

            # ── Interpretation ────────────────────────────────────────
            st.divider()
            st.subheader("💡 Stress-Test Interpretation")

            best = min(stress_results, key=lambda x: x.cvar_5_stressed)
            worst = max(stress_results, key=lambda x: x.prob_loss)

            st.markdown(f"""
**Szenario:** Marktschock **{market_shock}%** · Korrelation ≥ **{corr_stress:.2f}**
{f" · Sektorschock {shock_sector} **{sector_shock_pct}%**" if shock_sector != "Keiner" else ""}

- **Robustestes Portfolio** (niedrigstes CVaR): **{best.method_name}**
  – CVaR (5%) = {best.cvar_5_stressed:+.1%}
- **Höchste Verlustwahrscheinlichkeit**: **{worst.method_name}**
  – P(Verlust) = {worst.prob_loss:.1%}

> **Faustregel:** CVaR (5%) unter Stress > −40% → Konzentration
> reduzieren oder defensivere Assets beimischen.
""")

        render_stress_historical_section(pf, active)
        render_stress_macro_section(pf, active)
