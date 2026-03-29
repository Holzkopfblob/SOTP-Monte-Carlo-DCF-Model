"""Distribution analysis section for DCF results."""

from __future__ import annotations

import streamlit as st

from presentation.charts import cdf_plot, histogram_kde
from presentation.theme.tokens import CHART_COLORS


def render_distribution_section(results) -> None:
    """Render EV/Equity/Price distribution charts."""
    st.subheader("📈 Verteilungsanalyse")

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.plotly_chart(
            histogram_kde(
                results.total_ev,
                "Enterprise Value – Verteilung",
                "Enterprise Value (Mio.)",
            ),
            use_container_width=True,
        )
    with chart_c2:
        st.plotly_chart(
            histogram_kde(
                results.equity_values,
                "Equity Value – Verteilung",
                "Equity Value (Mio.)",
            ),
            use_container_width=True,
        )

    chart_c3, chart_c4 = st.columns(2)
    with chart_c3:
        st.plotly_chart(
            histogram_kde(
                results.price_per_share,
                "Verteilung – Preis je Aktie",
                "Preis je Aktie",
                color=CHART_COLORS["accent"],
            ),
            use_container_width=True,
        )
    with chart_c4:
        st.plotly_chart(
            cdf_plot(
                results.price_per_share,
                "CDF – Preis je Aktie",
                "Preis je Aktie",
                color=CHART_COLORS["accent"],
            ),
            use_container_width=True,
        )
