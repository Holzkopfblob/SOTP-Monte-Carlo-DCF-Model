"""Preset stress scenario section for portfolio stress page."""
from __future__ import annotations

import streamlit as st


def render_stress_preset_section(pf: dict) -> tuple[float, float, str, int]:
    """Render preset controls and return selected stress parameters."""
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

    sectors_in_portfolio = list(set(am.sector for am in pf["asset_metrics"]))
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
    return float(market_shock), float(corr_stress), shock_sector, int(sector_shock_pct)
