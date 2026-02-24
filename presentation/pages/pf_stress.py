"""
Portfolio Stress-Test Tab
=========================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.portfolio_service import PortfolioAnalyser
from application.portfolio_stress import PortfolioStressTester
from domain.portfolio_models import HISTORICAL_SCENARIOS, MACRO_SECTOR_SENSITIVITY
from presentation.charts import stress_comparison_chart
from presentation.pages.pf_common import active_results


def render_stress(tab) -> None:
    """Render Tab 5 (Stress-Tests & Szenario-Analyse)."""
    with tab:
        if st.session_state.pf_results is None:
            st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
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

        # ── Presets ───────────────────────────────────────────────────
        st.subheader("🎛️ Stress-Szenario konfigurieren")

        preset = st.selectbox(
            "Preset wählen",
            ["Benutzerdefiniert", "COVID-19 Crash", "GFC 2008", "Mild Correction"],
            key="stress_preset",
        )

        preset_defaults = {
            "COVID-19 Crash": (-35, 0.90, -20),
            "GFC 2008": (-50, 0.95, -40),
            "Mild Correction": (-15, 0.70, -10),
            "Benutzerdefiniert": (-30, 0.85, -20),
        }
        p_market, p_corr, p_sector = preset_defaults[preset]

        sc1, sc2 = st.columns(2)
        with sc1:
            market_shock = st.slider(
                "Marktschock (%)", min_value=-80, max_value=0,
                value=p_market,
            )
        with sc2:
            corr_stress = st.slider(
                "Korrelations-Stress (min ρ):",
                min_value=0.0, max_value=1.0, value=p_corr, step=0.05,
            )

        sectors_in_portfolio = list(set(
            am.sector for am in pf["asset_metrics"]
        ))
        shock_sector = st.selectbox(
            "Sektor-Schock (optional)",
            ["Keiner"] + sectors_in_portfolio,
        )
        sector_shock_pct = 0
        if shock_sector != "Keiner":
            sector_shock_pct = st.slider(
                f"Zusätzlicher Schock – {shock_sector} (%)",
                min_value=-80, max_value=0, value=p_sector,
            )

        st.divider()

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

            st.plotly_chart(
                stress_comparison_chart(
                    normal_port, stressed_port, sel_stress_method,
                ),
                use_container_width=True,
            )

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

        # ── Historical Scenario Comparison ────────────────────────────
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

        if st.button("📜 Alle historischen Szenarien berechnen",
                      use_container_width=True):
            tester = PortfolioStressTester(risk_free_rate=pf["rf"])
            portfolio_weights = {
                name: pr.weights for name, pr in active.items()
            }

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

                # Scenario description cards
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

        # ── Macro Factor Sensitivity ──────────────────────────────────
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
            names = [am.name for am in pf["asset_metrics"]]

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

            # Portfolio-level impact
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
