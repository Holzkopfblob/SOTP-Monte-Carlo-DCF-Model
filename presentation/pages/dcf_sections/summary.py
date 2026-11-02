"""Summary section renderers for DCF results."""

from __future__ import annotations

import streamlit as st

from presentation.layout.insights import render_summary_cards
from presentation.pages.dcf_sections.builders import build_key_metrics, build_stats_table


def format_mio(value: float) -> str:
    """Amounts in Mio.; switch to Mrd. from 10,000 Mio. so cards don't truncate."""
    if abs(value) >= 10_000:
        return f"{value / 1_000:,.1f} Mrd."
    return f"{value:,.0f} Mio."


def render_key_metrics_section(results) -> None:
    """Render top-row KPI cards for DCF results (median + range, not mean)."""
    key_metrics = build_key_metrics(results)
    ev_stats = key_metrics["ev"]
    eq_stats = key_metrics["equity"]
    ps_stats = key_metrics["price"]

    meta = (st.session_state.get("cfg_passthrough") or {}).get("metadata") or {}
    kurs = meta.get("kurs_bei_recherche")
    vs_kurs = (
        f"{ps_stats['Median'] / kurs - 1:+.0%} vs. Kurs {kurs:,.2f}"
        if isinstance(kurs, (int, float)) and kurs > 0 else None
    )
    render_summary_cards([
        ("Fairer Wert je Aktie (Median)", f"{ps_stats['Median']:,.2f}", vs_kurs),
        ("Bandbreite je Aktie (P5 – P95)",
         f"{ps_stats['P5 (5%)']:,.0f} – {ps_stats['P95 (95%)']:,.0f}", None),
        ("Equity Value (Median)", format_mio(eq_stats["Median"]), None),
        ("Enterprise Value (Median)", format_mio(ev_stats["Median"]), None),
    ])


def render_descriptive_stats_section(results) -> None:
    """Render descriptive statistics dataframe section."""
    st.subheader("📊 Deskriptive Statistiken")
    stats_df = build_stats_table(results)
    st.dataframe(
        stats_df.style.format("{:,.2f}"),
        width="stretch",
    )
