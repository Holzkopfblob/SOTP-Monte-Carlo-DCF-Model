"""Summary section renderers for DCF results."""

from __future__ import annotations

import streamlit as st

from presentation.layout.insights import render_summary_cards
from presentation.pages.dcf_sections.builders import build_key_metrics, build_stats_table


def render_key_metrics_section(results) -> None:
    """Render top-row KPI cards for DCF results."""
    key_metrics = build_key_metrics(results)
    ev_stats = key_metrics["ev"]
    eq_stats = key_metrics["equity"]
    ps_stats = key_metrics["price"]
    render_summary_cards([
        ("Ø Enterprise Value", f"{ev_stats['Mittelwert']:,.1f} Mio.", None),
        ("Ø Equity Value", f"{eq_stats['Mittelwert']:,.1f} Mio.", None),
        ("Ø Preis / Aktie", f"{ps_stats['Mittelwert']:,.2f}", None),
        ("Std.-Abw. Equity", f"{eq_stats['Std.-Abw.']:,.1f} Mio.", None),
    ])


def render_descriptive_stats_section(results) -> None:
    """Render descriptive statistics dataframe section."""
    st.subheader("📊 Deskriptive Statistiken")
    stats_df = build_stats_table(results)
    st.dataframe(
        stats_df.style.format("{:,.2f}"),
        use_container_width=True,
    )
