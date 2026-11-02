"""
DCF Simulation Tab – Launch the Monte-Carlo engine
====================================================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.models import (
    SamplingMethod,
    SegmentConfig,
    SimulationConfig,
)
from infrastructure.config_io import (
    RESULTS_FINGERPRINT_KEY,
    corporate_bridge_from_dists,
    split_bridge_param as _split_bridge_param,  # noqa: F401 – kept for callers/tests
    ui_fingerprint,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine

# setup-dict key (dcf_setup.render_setup) → bridge prefix (config_io)
_SETUP_BRIDGE_KEYS = {
    "bridge_corp_costs": "bridge_cc", "bridge_corp_discount": "bridge_cd",
    "bridge_net_debt": "bridge_nd", "bridge_shares": "bridge_sh",
    "bridge_minority": "bridge_mi", "bridge_pension": "bridge_pn",
    "bridge_non_op": "bridge_no", "bridge_associates": "bridge_as",
}


def render_simulation(
    container,
    setup: dict,
    segment_configs: list[SegmentConfig],
) -> None:
    """Render Simulation step.

    Reads setup parameters and segment configs, builds a
    :class:`SimulationConfig`, runs the MC engine and stores results
    in ``st.session_state``.
    """
    n_simulations = setup["n_simulations"]
    n_segments = setup["n_segments"]
    random_seed = setup["random_seed"]

    with container:
        st.header("Monte-Carlo-Simulation starten")
        st.caption("Prüfen Sie die Zusammenfassung und starten Sie dann die Berechnung. Anschließend wird der Schritt **📈 Ergebnisse** freigeschaltet.")

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
                width="stretch",
            )

        if run_button:
            if not segment_configs:
                st.error("Es sind keine Segmente konfiguriert. Gehen Sie zu **🏢 Segmente** und ergänzen Sie mindestens ein Segment.")
            else:
                # Same bridge assembly as headless runs (config_io)
                config = SimulationConfig(
                    n_simulations=n_simulations,
                    random_seed=random_seed,
                    segments=segment_configs,
                    corporate_bridge=corporate_bridge_from_dists(
                        {p: setup.get(k) for k, p in _SETUP_BRIDGE_KEYS.items()}
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
                # Ties the result snapshot in saved JSONs to exactly these inputs.
                st.session_state[RESULTS_FINGERPRINT_KEY] = ui_fingerprint(dict(st.session_state))
                progress_bar.progress(100, text="Fertig!")
                # The sidebar (status + download incl. result snapshot) was
                # rendered before this run finished – rerun so it is current.
                st.session_state["_sim_just_finished"] = n_simulations
                st.rerun()

        n_done = st.session_state.pop("_sim_just_finished", None)
        if n_done:
            st.success(f"✅ Simulation abgeschlossen – {n_done:,} Szenarien berechnet.")
            st.balloons()

        if st.session_state.results is not None:
            st.markdown("---")
            st.success("Simulation verfügbar. Wechseln Sie jetzt mit **Weiter: Ergebnisse** zum nächsten Schritt.")
