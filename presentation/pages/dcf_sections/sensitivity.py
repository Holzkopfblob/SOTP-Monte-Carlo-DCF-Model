"""Sensitivity section renderers for DCF results."""

from __future__ import annotations

import streamlit as st

from domain.statistics import compute_sensitivity, conditional_sensitivity
from presentation.charts import conditional_tornado_chart, tornado_chart


def render_sensitivity_section(results) -> None:
    """Render tornado chart based on global sensitivity."""
    st.subheader("🌪️ Sensitivitätsanalyse")
    st.caption(
        "Spearman-Rangkorrelation der stochastischen Inputvariablen "
        "mit dem Equity Value – zeigt die **Feature Importance** der "
        "Werttreiber."
    )

    sensitivities = compute_sensitivity(
        results.equity_values, results.input_samples
    )
    if sensitivities:
        st.plotly_chart(
            tornado_chart(sensitivities),
            use_container_width=True,
        )
    else:
        st.info(
            "Keine stochastischen Inputs vorhanden – alle Parameter "
            "sind deterministisch (fest). Setzen Sie mindestens einen "
            "Parameter auf eine Verteilung, um die Sensitivität zu sehen."
        )


def render_conditional_sensitivity_section(results) -> None:
    """Render bear vs bull conditional tornado chart."""
    if not results.input_samples:
        return

    st.subheader("🐻🐂 Conditional Sensitivity – Bear vs. Bull")
    st.caption(
        "Welche Inputvariablen treiben den Wert in schlechten (P<25 %) "
        "vs. guten (P>75 %) Szenarien? Unterschiedliche Treiber in Bear "
        "und Bull deuten auf nicht-lineare Zusammenhänge hin."
    )

    cond = conditional_sensitivity(
        results.equity_values, results.input_samples
    )
    bear = cond.get("bear", {})
    bull = cond.get("bull", {})

    if bear or bull:
        st.plotly_chart(
            conditional_tornado_chart(bear, bull, top_n=10),
            use_container_width=True,
        )
    else:
        st.info(
            "Keine stochastischen Inputs vorhanden – Conditional Sensitivity "
            "benötigt mindestens einen Verteilungs-Input."
        )
