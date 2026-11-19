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
from domain.checks import OK, config_checks
from infrastructure.monte_carlo_engine import MonteCarloEngine
from presentation.format import num

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
        st.caption("Prüfen Sie die Zusammenfassung und starten Sie dann die Berechnung. "
                   "Danach öffnen sich automatisch die **📈 Ergebnisse**.")

        st.info(
            f"**{n_segments} {'Segment' if n_segments == 1 else 'Segmente'}** · "
            f"{num(n_simulations)} Iterationen · Seed {random_seed}"
        )

        if segment_configs:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Segment": sc.name,
                        "Basisumsatz (Mio.)": num(sc.base_revenue, 1),
                        "Prognosejahre": sc.forecast_years,
                        "TV-Methode": sc.terminal_method.value,
                    }
                    for sc in segment_configs
                ]),
                hide_index=True, width="stretch",
            )

            for c in config_checks(segment_configs):
                if c.status != OK:
                    st.warning(f"{c.status} Check {c.id} · {c.name}: {c.detail}")

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
                # rendered before this run finished – rerun so it is current,
                # straight into the results step (toast there).
                st.session_state["_sim_just_finished"] = n_simulations
                st.session_state.wizard_step = "results"
                st.rerun()

        if st.session_state.results is not None:
            st.success("Für diese Eingaben liegt ein Ergebnis vor – **Weiter: Ergebnisse**.", icon="✅")
