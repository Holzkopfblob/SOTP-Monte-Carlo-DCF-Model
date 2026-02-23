"""
DCF Simulation Tab – Launch the Monte-Carlo engine
====================================================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.models import (
    CorporateBridgeConfig,
    SegmentConfig,
    SimulationConfig,
)
from application.simulation_service import SimulationService


def render_simulation(
    tab,
    setup: dict,
    segment_configs: list[SegmentConfig],
) -> None:
    """Render Tab 3 (Simulation).

    Reads setup parameters and segment configs, builds a
    :class:`SimulationConfig`, runs the MC engine and stores results
    in ``st.session_state``.
    """
    n_simulations = setup["n_simulations"]
    n_segments = setup["n_segments"]
    random_seed = setup["random_seed"]

    with tab:
        st.header("Monte-Carlo-Simulation starten")

        st.info(
            f"**Konfiguration:** {n_segments} Segment(e) · "
            f"{n_simulations:,} Iterationen · Seed {random_seed}"
        )

        if segment_configs:
            with st.expander("Segment-Übersicht", expanded=True):
                overview_rows = [
                    {
                        "Segment": sc.name,
                        "Basisumsatz": f"{sc.base_revenue:,.1f} Mio.",
                        "Prognosejahre": sc.forecast_years,
                        "TV-Methode": sc.terminal_method.value,
                    }
                    for sc in segment_configs
                ]
                st.table(pd.DataFrame(overview_rows))

        st.markdown("")
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            run_button = st.button(
                "🚀 Simulation starten",
                type="primary",
                use_container_width=True,
            )

        if run_button:
            if not segment_configs:
                st.error("Bitte konfigurieren Sie mindestens ein Segment.")
            else:
                config = SimulationConfig(
                    n_simulations=n_simulations,
                    random_seed=random_seed,
                    segments=segment_configs,
                    corporate_bridge=CorporateBridgeConfig(
                        annual_corporate_costs=setup["annual_corp_costs"],
                        corporate_cost_discount_rate=setup["corp_discount"] / 100.0,
                        net_debt=setup["net_debt"],
                        shares_outstanding=setup["shares"],
                        minority_interests=setup["minority_interests"],
                        pension_liabilities=setup["pension_liabilities"],
                        non_operating_assets=setup["non_operating_assets"],
                        associate_investments=setup["associate_investments"],
                        stochastic_corporate_costs=setup["stoch_corp_costs"],
                        stochastic_net_debt=setup["stoch_net_debt"],
                        stochastic_shares=setup["stoch_shares"],
                        stochastic_minority_interests=setup["stoch_minority"],
                        stochastic_pension_liabilities=setup["stoch_pension"],
                        stochastic_non_operating_assets=setup["stoch_non_op"],
                        stochastic_associate_investments=setup["stoch_associates"],
                    ),
                    mid_year_convention=setup["mid_year_conv"],
                    segment_correlation=setup.get("segment_correlation"),
                )

                progress_bar = st.progress(0, text="Initialisiere Simulation …")
                progress_bar.progress(10, text="Generiere stochastische Samples …")

                results = SimulationService.run_simulation(config)

                progress_bar.progress(90, text="Berechne Statistiken …")
                st.session_state.results = results
                st.session_state.config = config
                progress_bar.progress(100, text="Fertig!")

                st.success(
                    f"✅ Simulation abgeschlossen – "
                    f"{n_simulations:,} Szenarien berechnet."
                )
                st.balloons()

        if st.session_state.results is not None:
            st.markdown("---")
            st.markdown(
                "➡️ Wechseln Sie zum Tab **📈 Ergebnisse** "
                "für die vollständige Auswertung."
            )
