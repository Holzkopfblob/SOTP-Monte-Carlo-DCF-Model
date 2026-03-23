"""Quality and valuation diagnostics sections for DCF results."""

from __future__ import annotations

import numpy as np
import streamlit as st

from presentation.charts import (
    quality_score_breakdown_chart,
    quality_score_gauge,
    reinvestment_rate_chart,
    roic_histogram,
    roic_vs_wacc_scatter,
    tv_ev_decomposition_chart,
)
from presentation.pages.dcf_sections.builders import build_quality_payload


def render_tv_ev_section(results) -> None:
    """Show TV/EV decomposition per segment."""
    payload = build_quality_payload(results)
    tv_ev_ratios = payload.get("tv_ev_ratios")
    if not tv_ev_ratios:
        return

    st.subheader("🔍 TV / EV-Zerlegung")
    st.caption(
        "Anteil des Terminal Values am Enterprise Value je Segment. "
        "Werte **> 70 %** deuten darauf hin, dass die Bewertung stark "
        "von langfristigen Annahmen abhängt."
    )

    with st.expander("ℹ️ Warum ist TV/EV wichtig?", expanded=False):
        st.markdown(r"""
Der **Terminal Value (TV)** repräsentiert den Wert aller Cash Flows
*nach* dem expliziten Prognosezeitraum.  Je höher sein Anteil am
gesamten Enterprise Value, desto mehr hängt die Bewertung von der
Wachstums- und WACC-Annahme im Restwert ab.

$$\text{TV/EV} = \frac{PV(\text{TV})}{PV(\text{FCFF}) + PV(\text{TV})}$$

| TV/EV | Einschätzung |
|-------|-------------|
| < 50 % | Robust – Großteil des Werts fällt in den Prognosezeitraum |
| 50–70 % | Typisch für viele Branchen |
| > 70 % | Fragil – empfindlich gegenüber Terminal-Growth & WACC |
""")

    seg_names: list[str] = []
    tv_shares: list[float] = []
    fcff_shares: list[float] = []

    for seg_name, tv_ev_arr in tv_ev_ratios.items():
        mean_tv = float(np.mean(tv_ev_arr))
        seg_names.append(seg_name)
        tv_shares.append(mean_tv)
        fcff_shares.append(1.0 - mean_tv)

    st.plotly_chart(
        tv_ev_decomposition_chart(seg_names, fcff_shares, tv_shares),
        use_container_width=True,
    )

    cols = st.columns(len(seg_names))
    for col, name, tv in zip(cols, seg_names, tv_shares):
        label = "🟢" if tv < 0.50 else ("🟡" if tv < 0.70 else "🔴")
        col.metric(f"TV/EV – {name}", f"{tv:.1%}", delta=label, delta_color="off")


def render_quality_section(results) -> None:
    """Show composite valuation quality score."""
    score = build_quality_payload(results).get("score")
    if not score:
        return

    st.subheader("🏅 Bewertungsqualität")
    st.caption(
        "Composite-Score (0 – 100) aggregiert vier Dimensionen: "
        "TV/EV-Risiko, Konvergenz, Sensitivitäts-Diversifikation, "
        "Ergebnis-Streuung."
    )

    with st.expander("ℹ️ Wie wird der Score berechnet?", expanded=False):
        st.markdown("""
| Dimension (je max 25 Pkt.) | Gut | Schlecht |
|---|---|---|
| **TV/EV Risiko** | TV/EV ≤ 40 % | TV/EV ≥ 90 % |
| **Konvergenz** | KI-Breite < 0.5 % | KI-Breite > 5 % |
| **Sensitivitäts-Diversifikation** | Viele gleichwichtige Treiber | Ein Treiber dominiert |
| **Ergebnis-Streuung** | CV < 0.1 | CV > 1.0 |

**Interpretation:**
- **70–100**: Hohe Bewertungsqualität – robuste Ergebnisse
- **40–70**: Akzeptabel – prüfen Sie die schwächsten Dimensionen
- **< 40**: Niedrig – Ergebnisse sind fragil, Annahmen überprüfen
""")

    qc1, qc2 = st.columns([1, 1])
    with qc1:
        st.plotly_chart(
            quality_score_gauge(score),
            use_container_width=True,
        )
    with qc2:
        st.plotly_chart(
            quality_score_breakdown_chart(score),
            use_container_width=True,
        )


def render_roic_section(results, config) -> None:
    """Show implied ROIC and reinvestment rate per segment."""
    if not results.segment_implied_roic:
        return

    st.subheader("📊 Implied ROIC & Reinvestitionsrate")
    st.caption(
        "Implizierter Return on Invested Capital, abgeleitet aus den "
        "Modellannahmen (Marge, CAPEX, NWC, Wachstum). "
        "Vergleichen Sie mit dem historischen ROIC des Unternehmens "
        "als Plausibilitäts-Check."
    )

    with st.expander("ℹ️ Was zeigt die Implied ROIC?", expanded=False):
        st.markdown(r"""
Der **Implied ROIC** wird nicht direkt modelliert, sondern ergibt sich
*implizit* aus den Value-Driver-Annahmen über die Steady-State-Identität
$g = \text{ROIC} \times b$:

$$\text{NOPAT-Marge} = (\text{EBITDA\%} - \text{D\&A\%}) \times (1 - t)$$

$$b = \frac{\text{CAPEX\%} - \text{D\&A\%} + \text{NWC\%} \times \frac{g}{1+g}}{\text{NOPAT-Marge}}$$

$$\text{Implied ROIC} = \frac{g}{b} = g \times \frac{\text{NOPAT-Marge}}{\text{Reinvest.-Marge}}$$

| ROIC vs. WACC | Bedeutung |
|---|---|
| **ROIC > WACC** | Wertschöpfung – das Segment erwirtschaftet mehr als die Kapitalkosten |
| **ROIC ≈ WACC** | Weder Wert geschaffen noch vernichtet |
| **ROIC < WACC** | Wertvernichtung – Kapitalkosten werden nicht gedeckt |

> **Tipp:** Wenn der Implied ROIC deutlich über dem historischen ROIC liegt,
sind die Annahmen möglicherweise zu optimistisch.
""")

    seg_wacc: dict[str, np.ndarray] = {}
    weighted_wacc = 0.0
    total_ev_sum = 0.0
    for seg in config.segments:
        seg_key = f"{seg.name} | WACC"
        if seg_key in results.input_samples:
            seg_wacc[seg.name] = results.input_samples[seg_key]
            mean_wacc = float(np.mean(results.input_samples[seg_key]))
            mean_ev = float(np.mean(results.segment_evs.get(seg.name, np.array([0]))))
            weighted_wacc += mean_wacc * mean_ev
            total_ev_sum += mean_ev

    avg_wacc = weighted_wacc / max(total_ev_sum, 1e-6)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(
            roic_histogram(results.segment_implied_roic, wacc_mean=avg_wacc),
            use_container_width=True,
        )

    with rc2:
        st.plotly_chart(
            reinvestment_rate_chart(results.segment_reinvest_rates),
            use_container_width=True,
        )

    if seg_wacc:
        st.plotly_chart(
            roic_vs_wacc_scatter(results.segment_implied_roic, seg_wacc),
            use_container_width=True,
        )

    cols = st.columns(len(results.segment_implied_roic))
    for col, (seg_name, roic_arr) in zip(cols, results.segment_implied_roic.items()):
        mean_roic = float(np.mean(roic_arr))
        wacc_ref = float(np.mean(seg_wacc.get(seg_name, np.array([avg_wacc]))))
        spread = mean_roic - wacc_ref
        label = "🟢" if spread > 0.02 else ("🟡" if spread > -0.02 else "🔴")
        col.metric(
            f"ROIC – {seg_name}",
            f"{mean_roic:.1%}",
            delta=f"{spread:+.1%} vs WACC {label}",
            delta_color="normal",
        )
