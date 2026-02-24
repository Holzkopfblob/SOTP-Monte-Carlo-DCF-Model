"""
DCF Simulation Tab – Launch the Monte-Carlo engine
====================================================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    DistributionType,
    SamplingMethod,
    SegmentConfig,
    SimulationConfig,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine


def _split_bridge_param(
    dist: DistributionConfig | None,
    fallback: float = 0.0,
) -> tuple[float, DistributionConfig | None]:
    """Extract scalar + optional stochastic override from a unified config.

    When the distribution is "Fest", the scalar value is used and no
    stochastic override is passed to the engine.  For any other type
    the representative value becomes the scalar fallback while the full
    distribution is forwarded for MC sampling.
    """
    if dist is None:
        return fallback, None
    if dist.dist_type == DistributionType.FIXED:
        return dist.fixed_value, None
    return dist.representative_value(), dist


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
                # Split unified DistributionConfig → scalar + stochastic
                cc_val, cc_stoch = _split_bridge_param(setup["bridge_corp_costs"], 50.0)
                cd_val, cd_stoch = _split_bridge_param(setup["bridge_corp_discount"], 0.09)
                nd_val, nd_stoch = _split_bridge_param(setup["bridge_net_debt"], 500.0)
                sh_val, sh_stoch = _split_bridge_param(setup["bridge_shares"], 100.0)
                mi_val, mi_stoch = _split_bridge_param(setup.get("bridge_minority"))
                pn_val, pn_stoch = _split_bridge_param(setup.get("bridge_pension"))
                no_val, no_stoch = _split_bridge_param(setup.get("bridge_non_op"))
                as_val, as_stoch = _split_bridge_param(setup.get("bridge_associates"))

                config = SimulationConfig(
                    n_simulations=n_simulations,
                    random_seed=random_seed,
                    segments=segment_configs,
                    corporate_bridge=CorporateBridgeConfig(
                        annual_corporate_costs=cc_val,
                        corporate_cost_discount_rate=cd_val,
                        net_debt=nd_val,
                        shares_outstanding=sh_val,
                        minority_interests=mi_val,
                        pension_liabilities=pn_val,
                        non_operating_assets=no_val,
                        associate_investments=as_val,
                        stochastic_corporate_costs=cc_stoch,
                        stochastic_corporate_cost_discount_rate=cd_stoch,
                        stochastic_net_debt=nd_stoch,
                        stochastic_shares=sh_stoch,
                        stochastic_minority_interests=mi_stoch,
                        stochastic_pension_liabilities=pn_stoch,
                        stochastic_non_operating_assets=no_stoch,
                        stochastic_associate_investments=as_stoch,
                    ),
                    mid_year_convention=setup["mid_year_conv"],
                    sampling_method=setup.get("sampling_method", SamplingMethod.PSEUDO_RANDOM),
                    segment_correlation=setup.get("segment_correlation"),
                )

                progress_bar = st.progress(0, text="Initialisiere Simulation …")
                progress_bar.progress(10, text="Generiere stochastische Samples …")

                results = MonteCarloEngine(config).run()

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
