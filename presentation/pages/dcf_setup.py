"""
DCF Setup Tab – Global simulation parameters & corporate bridge
================================================================
"""
from __future__ import annotations

import streamlit as st

from presentation.pages.dcf_sections.setup_bridge import render_setup_bridge_section
from presentation.pages.dcf_sections.setup_correlation import (
    render_setup_correlation_section,
)
from presentation.pages.dcf_sections.setup_simulation import (
    render_setup_simulation_section,
)
from presentation.ui_helpers import (
    render_info_corporate_bridge,
    render_info_monte_carlo,
    render_info_sotp,
)


def render_setup(container) -> dict:
    """Render Setup step and return all configuration values as a dict.

    The returned dict contains every scalar/distribution needed by downstream
    tabs (segments, simulation).
    """
    with container:
        st.header("Modell-Konfiguration")
        render_info_sotp()
        render_info_monte_carlo()
        sim_cfg = render_setup_simulation_section()

        # ── Corporate bridge ──────────────────────────────────────────
        render_info_corporate_bridge()
        bridge_cfg = render_setup_bridge_section()

        # ── Cross-segment correlation (Phase 3) ──────────────────────
        segment_correlation = render_setup_correlation_section(sim_cfg["n_segments"])

    return {
        "n_simulations": sim_cfg["n_simulations"],
        "random_seed": sim_cfg["random_seed"],
        "n_segments": sim_cfg["n_segments"],
        "mid_year_conv": sim_cfg["mid_year_conv"],
        "sampling_method": sim_cfg["sampling_method"],
        "bridge_corp_costs": bridge_cfg["bridge_corp_costs"],
        "bridge_corp_discount": bridge_cfg["bridge_corp_discount"],
        "bridge_net_debt": bridge_cfg["bridge_net_debt"],
        "bridge_shares": bridge_cfg["bridge_shares"],
        "bridge_minority": bridge_cfg["bridge_minority"],
        "bridge_pension": bridge_cfg["bridge_pension"],
        "bridge_non_op": bridge_cfg["bridge_non_op"],
        "bridge_associates": bridge_cfg["bridge_associates"],
        "segment_correlation": segment_correlation,
    }
