"""Simulation controls section for DCF setup page."""

from __future__ import annotations

import streamlit as st

from domain.models import SamplingMethod


def render_setup_simulation_section() -> dict:
    """Render global simulation controls and return normalized values."""
    col_a, col_b = st.columns(2)
    with col_a:
        n_simulations = st.number_input(
            "Anzahl Monte-Carlo-Iterationen",
            min_value=1_000, max_value=500_000, value=10_000, step=1_000,
            help="Mehr Iterationen -> genauere Ergebnisse, längere Laufzeit.",
            key="setup_n_sim",
        )
        random_seed = st.number_input(
            "Random Seed (Reproduzierbarkeit)",
            value=42, min_value=0, key="setup_seed",
        )
    with col_b:
        n_segments = st.number_input(
            "Anzahl Geschäftssegmente",
            min_value=1, max_value=20, value=2,
            help="Für jedes Segment wird ein separater DCF berechnet.",
            key="setup_n_seg",
        )

    col_c, col_d = st.columns(2)
    with col_c:
        sampling_label = st.selectbox(
            "Sampling-Methode (Varianzreduktion)",
            [s.value for s in SamplingMethod],
            index=0,
            key="setup_sampling",
            help="Antithetic Variates halbiert die Varianz bei gleicher "
                 "Iterationsanzahl. Sobol (Quasi-MC) erzeugt gleichmäßigere "
                 "Abtastung des Parameterraums.",
        )
        sampling_method = SamplingMethod(sampling_label)

    _ = col_d
    st.markdown("")
    mid_year_conv = st.checkbox(
        "⏱️ Mid-Year Discounting Convention",
        value=True, key="setup_mid_year",
        help="Diskontiert FCFFs zur Jahresmitte (t−0,5) statt zum Jahresende. "
             "Standardpraxis bei DCF-Bewertungen, da Cashflows unterjährig anfallen.",
    )

    st.divider()
    return {
        "n_simulations": int(n_simulations),
        "random_seed": int(random_seed),
        "n_segments": int(n_segments),
        "sampling_method": sampling_method,
        "mid_year_conv": bool(mid_year_conv),
    }
