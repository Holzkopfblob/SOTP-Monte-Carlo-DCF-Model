"""Summary section renderers for DCF results."""

from __future__ import annotations

import streamlit as st

from presentation.format import mio, num, pct
from presentation.layout.insights import render_summary_cards
from presentation.pages.dcf_sections.builders import build_key_metrics, build_stats_table


def render_key_metrics_section(results) -> None:
    """Render top-row KPI cards for DCF results (median + range, not mean)."""
    key_metrics = build_key_metrics(results)
    ev_stats = key_metrics["ev"]
    eq_stats = key_metrics["equity"]
    ps_stats = key_metrics["price"]

    meta = (st.session_state.get("cfg_passthrough") or {}).get("metadata") or {}
    kurs = meta.get("kurs_bei_recherche")
    vs_kurs = (
        f"{pct(ps_stats['Median'] / kurs - 1, 0, signed=True)} vs. Kurs {num(kurs, 2)}"
        if isinstance(kurs, (int, float)) and kurs > 0 else None
    )
    render_summary_cards([
        ("Fairer Wert je Aktie (Median)", num(ps_stats['Median'], 2), vs_kurs),
        ("Bandbreite je Aktie (P5 – P95)",
         f"{num(ps_stats['P5 (5%)'])} – {num(ps_stats['P95 (95%)'])}", None),
        ("Equity Value (Median)", mio(eq_stats["Median"]), None),
        ("Enterprise Value (Median)", mio(ev_stats["Median"]), None),
    ])


def render_descriptive_stats_section(results) -> None:
    """Render descriptive statistics dataframe section."""
    st.subheader("📊 Deskriptive Statistiken")
    stats_df = build_stats_table(results)
    st.caption("Beträge in Mio., Preis je Aktie in Berichtswährung; Schiefe, Kurtosis und CV dimensionslos.")
    st.dataframe(
        stats_df.style.format(lambda v: num(v, 2)),
        width="stretch",
    )
