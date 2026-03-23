"""Risk-oriented sections for DCF results."""

from __future__ import annotations

import numpy as np
import streamlit as st

from presentation.charts import (
    economic_profit_chart,
    implied_return_cdf,
    margin_of_safety_chart,
    percentile_convergence_chart,
)
from presentation.pages.dcf_sections.builders import build_tail_risk_metrics


def render_tail_risk_section(results) -> None:
    """Show VaR, CVaR, tail ratio and percentile convergence chart."""
    st.subheader("⚠️ Tail-Risiko & Perzentil-Konvergenz")
    st.caption(
        "Value-at-Risk (VaR) und Conditional VaR (CVaR) quantifizieren "
        "das Downside-Risiko. Die Perzentil-Konvergenz zeigt, ob P5/P50/P95 "
        "sich stabilisiert haben."
    )

    with st.expander("ℹ️ Was bedeuten VaR, CVaR und Tail Ratio?", expanded=False):
        st.markdown(r"""
| Kennzahl | Definition |
|---|---|
| **VaR (5 %)** | Equity Value, unter den nur **5 %** der Szenarien fallen |
| **CVaR (5 %)** | Durchschnitt aller Szenarien *unterhalb* des VaR – erfasst die Schwere der schlimmsten Fälle |
| **Tail Ratio** | VaR / CVaR – je näher an 1,0, desto kürzer der linke Tail |

$$\text{CVaR}_\alpha = \mathbb{E}[X \mid X \leq \text{VaR}_\alpha]$$

> Ein **Tail Ratio < 0,8** deutet auf einen langen, schweren linken
> Tail hin — die schlimmsten 5 % der Szenarien sind deutlich schlechter
> als der VaR allein suggeriert.
""")

    metrics = build_tail_risk_metrics(results)
    var_val = metrics["var_5"]
    cvar_val = metrics["cvar_5"]
    tail_ratio = metrics["tail_ratio"]

    if var_val is not None and cvar_val is not None:
        tr1, tr2, tr3 = st.columns(3)
        tr1.metric("VaR 5 % (Equity)", f"{var_val:,.1f} Mio.")
        tr2.metric("CVaR 5 % (Equity)", f"{cvar_val:,.1f} Mio.")
        if tail_ratio is not None and tail_ratio != 0:
            label = "🟢" if tail_ratio > 0.85 else ("🟡" if tail_ratio > 0.70 else "🔴")
            tr3.metric("Tail Ratio", f"{tail_ratio:.3f}", delta=label, delta_color="off")
        else:
            tr3.metric("Tail Ratio", "n/a")

    p5 = getattr(results, "convergence_p5", None)
    p50 = getattr(results, "convergence_p50", None)
    p95 = getattr(results, "convergence_p95", None)

    if (
        p5 is not None
        and p50 is not None
        and p95 is not None
        and len(results.convergence_indices) > 0
    ):
        st.plotly_chart(
            percentile_convergence_chart(
                results.convergence_indices, p5, p50, p95,
            ),
            use_container_width=True,
        )


def render_economic_profit_section(results) -> None:
    """Economic Profit (EVA) per segment + probability of value destruction."""
    seg_ep = getattr(results, "segment_economic_profit", None)
    seg_pvd = getattr(results, "segment_prob_value_destruction", None)

    if not seg_ep:
        return

    st.subheader("💰 Economic Profit (EVA) je Segment")
    st.caption(
        "Der Economic Profit zeigt, ob ein Segment mehr Rendite erwirtschaftet "
        "als die Kapitalkosten betragen. EP < 0 bedeutet Wertvernichtung."
    )

    with st.expander("ℹ️ Was ist Economic Profit?", expanded=False):
        st.markdown(r"""
$$\text{EP} = \text{NOPAT} - \text{WACC} \times \text{Invested Capital}$$

Alternativ geschätzt über den Value-Driver-Ansatz:

$$\text{EP} \approx \text{Revenue} \times [(\text{EBITDA\%} - \text{D\&A\%}) \times (1 - t)
- (\text{CAPEX\%} - \text{D\&A\%} + \Delta\text{NWC\%}) \times \frac{g}{1+g}]
- \text{WACC} \times \frac{\text{FCF}}{(\text{WACC} - g)}$$

> **Interpretation:** EP > 0 bedeutet wirtschaftliche Wertschöpfung
> (ROIC > WACC). P(EP < 0) quantifiziert die Wahrscheinlichkeit,
> dass ein Segment seine Kapitalkosten *nicht* deckt.
""")

    st.plotly_chart(
        economic_profit_chart(seg_ep),
        use_container_width=True,
    )

    if seg_pvd:
        cols = st.columns(len(seg_pvd))
        for col, (seg_name, pvd) in zip(cols, seg_pvd.items()):
            label = "🟢" if pvd < 0.15 else ("🟡" if pvd < 0.35 else "🔴")
            col.metric(
                f"P(ROIC < WACC) – {seg_name}",
                f"{pvd:.1%}",
                delta=label,
                delta_color="off",
            )


def render_margin_of_safety_section(results) -> None:
    """Interactive MoS analysis with market price input."""
    st.subheader("🛡️ Margin-of-Safety Analyse")
    st.caption(
        "Geben Sie den aktuellen Markt- oder Kaufkurs ein, um die implizierte "
        "Rendite, die Wahrscheinlichkeit einer Unterbewertung und den "
        "empfohlenen Kaufpreis mit Sicherheitsmarge zu berechnen."
    )

    prices = results.price_per_share
    fair_mean = float(np.mean(prices))

    market_price = st.number_input(
        "Aktueller Marktpreis / Kurs",
        min_value=0.01,
        value=round(fair_mean * 0.9, 2),
        step=0.5,
        format="%.2f",
        help="Der Preis, zu dem die Aktie aktuell gehandelt wird.",
    )

    if market_price > 0:
        p_upside = float(np.mean(prices >= market_price))
        implied_return = (fair_mean / market_price - 1.0) * 100
        buy_price_30 = fair_mean * 0.70

        mo1, mo2, mo3, mo4 = st.columns(4)
        mo1.metric("Ø Fair Value", f"{fair_mean:,.2f}")
        mo2.metric("P(Upside)", f"{p_upside:.1%}")
        mo3.metric("Impl. Rendite", f"{implied_return:+.1f} %")
        mo4.metric("Kaufkurs (30 % MoS)", f"{buy_price_30:,.2f}")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.plotly_chart(
                margin_of_safety_chart(prices, market_price),
                use_container_width=True,
            )
        with mc2:
            st.plotly_chart(
                implied_return_cdf(prices, market_price),
                use_container_width=True,
            )
