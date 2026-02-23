"""
DCF Results Tab – Interactive charts, statistics & Excel export
================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import skew, kurtosis

from domain.statistics import compute_statistics
from application.simulation_service import SimulationService
from infrastructure.excel_export import ExcelExporter
from presentation.ui_helpers import render_info_interpretation
from presentation.charts import (
    cdf_plot,
    convergence_chart,
    histogram_kde,
    implied_roic_chart,
    quality_score_breakdown_chart,
    quality_score_gauge,
    tornado_chart,
    tv_ev_decomposition_chart,
    waterfall_chart,
)


def render_results(tab) -> None:
    """Render Tab 4 (Ergebnisse).

    Reads ``st.session_state.results`` and ``st.session_state.config``.
    """
    with tab:
        if st.session_state.results is None:
            st.warning(
                "⚠️ Bitte führen Sie zunächst eine Simulation durch "
                "(Tab **🎲 Simulation**)."
            )
            return

        results = st.session_state.results
        config = st.session_state.config

        st.header("📈 Simulationsergebnisse")
        render_info_interpretation()

        # ── Key metrics row ───────────────────────────────────────────
        ev_stats = compute_statistics(results.total_ev)
        eq_stats = compute_statistics(results.equity_values)
        ps_stats = compute_statistics(results.price_per_share)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ø Enterprise Value", f"{ev_stats['Mittelwert']:,.1f} Mio.")
        m2.metric("Ø Equity Value", f"{eq_stats['Mittelwert']:,.1f} Mio.")
        m3.metric("Ø Preis / Aktie", f"{ps_stats['Mittelwert']:,.2f}")
        m4.metric("Std.-Abw. Equity", f"{eq_stats['Std.-Abw.']:,.1f} Mio.")

        st.divider()

        # ── Descriptive statistics table ──────────────────────────────
        st.subheader("📊 Deskriptive Statistiken")

        stats_data = {
            "Enterprise Value": ev_stats,
            "Equity Value": eq_stats,
            "Preis / Aktie": ps_stats,
        }
        for seg_name, seg_ev in results.segment_evs.items():
            stats_data[f"EV – {seg_name}"] = compute_statistics(seg_ev)

        stats_df = pd.DataFrame(stats_data).T
        st.dataframe(
            stats_df.style.format("{:,.2f}"),
            use_container_width=True,
        )

        st.divider()

        # ── Distribution charts ───────────────────────────────────────
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
                cdf_plot(
                    results.equity_values,
                    "CDF – Equity Value",
                    "Equity Value (Mio.)",
                ),
                use_container_width=True,
            )
        with chart_c4:
            st.plotly_chart(
                histogram_kde(
                    results.price_per_share,
                    "Verteilung – Preis je Aktie",
                    "Preis je Aktie",
                    color="#9467bd",
                ),
                use_container_width=True,
            )

        st.divider()

        # ── Tornado chart ─────────────────────────────────────────────
        st.subheader("🌪️ Sensitivitätsanalyse")
        st.caption(
            "Spearman-Rangkorrelation der stochastischen Inputvariablen "
            "mit dem Equity Value – zeigt die **Feature Importance** der "
            "Werttreiber."
        )

        sensitivities = SimulationService.compute_sensitivity(results)
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

        st.divider()

        # ── Waterfall chart ───────────────────────────────────────────
        st.subheader("🏗️ SOTP-Wertbrücke")
        st.caption(
            "Erwartungswerte (Ø) der einzelnen Segmente, abzüglich "
            "Holdingkosten und Nettoverschuldung."
        )

        st.plotly_chart(
            waterfall_chart(
                results.base_segment_evs,
                results.base_corporate_costs_pv,
                results.base_net_debt,
                results.base_equity_value,
                minority_interests=results.base_minority_interests,
                pension_liabilities=results.base_pension_liabilities,
                non_operating_assets=results.base_non_operating_assets,
                associate_investments=results.base_associate_investments,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Phase 2: TV/EV Decomposition ──────────────────────────────
        _render_tv_ev_section(results)

        st.divider()

        # ── Phase 2: Implied ROIC ─────────────────────────────────────
        _render_roic_section(results)

        st.divider()

        # ── Phase 2: Quality Score ────────────────────────────────────
        _render_quality_section(results)

        st.divider()

        # ── Convergence diagnostics ───────────────────────────────────
        st.subheader("🔬 Konvergenz-Diagnose")
        st.caption(
            "Zeigt, ob die Anzahl der Simulationen ausreicht. "
            "Wenn der laufende Mittelwert sich stabilisiert und das "
            "95 %-Konfidenzintervall eng wird, sind die Ergebnisse konvergiert."
        )

        if len(results.convergence_indices) > 0:
            st.plotly_chart(
                convergence_chart(
                    results.convergence_indices,
                    results.convergence_means,
                    results.convergence_ci_low,
                    results.convergence_ci_high,
                ),
                use_container_width=True,
            )

            final_width = (
                results.convergence_ci_high[-1] - results.convergence_ci_low[-1]
            )
            final_mean = results.convergence_means[-1]
            pct_width = (
                (final_width / abs(final_mean) * 100) if abs(final_mean) > 0 else 0
            )

            conv_c1, conv_c2, conv_c3 = st.columns(3)
            conv_c1.metric("KI-Breite (absolut)", f"{final_width:,.1f} Mio.")
            conv_c2.metric("KI-Breite (relativ)", f"{pct_width:.3f} %")

            if pct_width < 0.5:
                conv_c3.metric("Status", "✅ Konvergiert")
                st.success(
                    f"Die Simulation ist gut konvergiert. Das 95 %-Konfidenzintervall "
                    f"beträgt nur **{pct_width:.3f} %** des Mittelwerts."
                )
            elif pct_width < 2.0:
                conv_c3.metric("Status", "⚠️ Akzeptabel")
                st.warning(
                    f"Die Konvergenz ist akzeptabel ({pct_width:.2f} %), "
                    f"aber eine Erhöhung der Iterationszahl könnte die Stabilität verbessern."
                )
            else:
                conv_c3.metric("Status", "❌ Nicht konvergiert")
                st.error(
                    f"Die Ergebnisse sind noch nicht stabil ({pct_width:.1f} %). "
                    f"Erhöhen Sie die Anzahl der Iterationen deutlich (mindestens 2–3×)."
                )

        st.divider()

        # ── Per-segment detail ────────────────────────────────────────
        if len(results.segment_evs) > 1:
            st.subheader("📦 Segment-Details")
            seg_tabs = st.tabs(list(results.segment_evs.keys()))
            for stab, (seg_name, seg_ev) in zip(
                seg_tabs, results.segment_evs.items()
            ):
                with stab:
                    st.plotly_chart(
                        histogram_kde(
                            seg_ev,
                            f"EV-Verteilung – {seg_name}",
                            "Enterprise Value (Mio.)",
                        ),
                        use_container_width=True,
                    )
            st.divider()

        # ── Portfolio app parameters ──────────────────────────────────
        _render_portfolio_params(results)

        st.divider()

        # ── Excel export ──────────────────────────────────────────────
        st.subheader("📥 Excel-Export")
        st.markdown(
            "Der Report enthält drei Arbeitsblätter: "
            "**Summary & Statistics**, **Segment Assumptions** "
            "und **Raw Simulation Data**."
        )

        excel_bytes = ExcelExporter(config, results).generate()
        st.download_button(
            label="📥 Vollständigen Excel-Report herunterladen",
            data=excel_bytes,
            file_name="sotp_mc_dcf_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )


# ── Private helper ────────────────────────────────────────────────────────

def _render_portfolio_params(results) -> None:
    """Show distribution parameters for handoff to the portfolio app."""
    st.subheader("🔗 Verteilungsparameter für Portfolio-App")
    st.markdown(
        "Übertragen Sie diese Werte in die **Portfolio-Optimierung** "
        "(`portfolio_app.py`), um die simulierte Fair-Value-Verteilung "
        "dort als Input zu nutzen."
    )

    with st.expander("ℹ️ Wie übertrage ich die Werte?", expanded=False):
        st.markdown("""
### So nutzen Sie die Parameter in der Portfolio-App

1. Notieren Sie sich die **5 Kennzahlen** unten (μ, σ, Schiefe, P5, P95)
2. Öffnen Sie die **Portfolio-App** (`streamlit run portfolio_app.py --server.port 8502`)
3. Wählen Sie als Verteilungstyp **"Aus DCF-App (μ, σ, Schiefe)"**
4. Geben Sie die Werte ein – die App rekonstruiert automatisch die passende Verteilung:
   - **Schiefe ≈ 0** → Normalverteilung (symmetrisch)
   - **Schiefe > 0** → Lognormalverteilung (rechtsschiefe MC-Ergebnisse)
5. Die Portfolio-App generiert daraus eine Fair-Value-Verteilung, die der
   MC-Simulation möglichst nahekommt.

> **Tipp:** Für jedes Unternehmen, das Sie per SOTP-DCF bewertet haben,
> können Sie die Parameter übertragen und so ein **Multi-Aktien-Portfolio**
> optimieren.
""")

    prices = results.price_per_share
    p_mean = float(np.mean(prices))
    p_std = float(np.std(prices))
    p_median = float(np.median(prices))
    p_skew = float(skew(prices))
    p_kurt = float(kurtosis(prices))
    p_p5 = float(np.percentile(prices, 5))
    p_p25 = float(np.percentile(prices, 25))
    p_p75 = float(np.percentile(prices, 75))
    p_p95 = float(np.percentile(prices, 95))

    st.markdown("##### 📋 Parameter zum Übertragen")

    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("μ (Mittelwert)", f"{p_mean:,.2f}")
    pk2.metric("σ (Std.-Abw.)", f"{p_std:,.2f}")
    pk3.metric("Schiefe (Skew)", f"{p_skew:,.3f}")
    pk4.metric("P5", f"{p_p5:,.2f}")
    pk5.metric("P95", f"{p_p95:,.2f}")

    pk6, pk7, pk8, pk9, _ = st.columns(5)
    pk6.metric("Median (P50)", f"{p_median:,.2f}")
    pk7.metric("P25", f"{p_p25:,.2f}")
    pk8.metric("P75", f"{p_p75:,.2f}")
    pk9.metric("Kurtosis", f"{p_kurt:,.3f}")

    if abs(p_skew) < 0.5:
        rec_dist = "Normal"
        st.success(
            f"📊 **Empfehlung: Normalverteilung** (Schiefe = {p_skew:,.3f} ≈ 0) · "
            f"Geben Sie in der Portfolio-App ein: **μ = {p_mean:,.2f}** · **σ = {p_std:,.2f}**"
        )
    else:
        rec_dist = "Lognormal"
        st.info(
            f"📊 **Empfehlung: Lognormalverteilung** (Schiefe = {p_skew:,.3f} ≠ 0) · "
            f"Geben Sie in der Portfolio-App ein: **μ = {p_mean:,.2f}** · **σ = {p_std:,.2f}** · "
            f"**Schiefe = {p_skew:,.3f}**"
        )

    st.markdown("##### 📎 Kopiervorlage")
    st.code(
        f"Verteilungstyp: Aus DCF-App (μ, σ, Schiefe)\n"
        f"μ (Mittelwert):  {p_mean:,.4f}\n"
        f"σ (Std.-Abw.):   {p_std:,.4f}\n"
        f"Schiefe (Skew):  {p_skew:,.4f}\n"
        f"─────────────────────────────\n"
        f"Empf. Verteilung: {rec_dist}\n"
        f"Median:           {p_median:,.4f}\n"
        f"P5 / P95:         {p_p5:,.4f} / {p_p95:,.4f}\n"
        f"Kurtosis:         {p_kurt:,.4f}\n"
        f"Simulationen:     {results.n_simulations:,}",
        language="text",
    )


# ── Phase 2: TV/EV section ───────────────────────────────────────────────

def _render_tv_ev_section(results) -> None:
    """Show TV/EV decomposition per segment."""
    if not results.segment_tv_ev_ratios:
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

    for seg_name, tv_ev_arr in results.segment_tv_ev_ratios.items():
        mean_tv = float(np.mean(tv_ev_arr))
        seg_names.append(seg_name)
        tv_shares.append(mean_tv)
        fcff_shares.append(1.0 - mean_tv)

    st.plotly_chart(
        tv_ev_decomposition_chart(seg_names, fcff_shares, tv_shares),
        use_container_width=True,
    )

    # Metric tiles
    cols = st.columns(len(seg_names))
    for col, name, tv in zip(cols, seg_names, tv_shares):
        label = "🟢" if tv < 0.50 else ("🟡" if tv < 0.70 else "🔴")
        col.metric(f"TV/EV – {name}", f"{tv:.1%}", delta=label,
                   delta_color="off")


# ── Phase 2: Implied ROIC section ────────────────────────────────────────

def _render_roic_section(results) -> None:
    """Show implied ROIC per segment."""
    if not results.segment_implied_roic:
        return

    st.subheader("💰 Implizierte ROIC")
    st.caption(
        "Return on Invested Capital, impliziert durch die Modell-Annahmen "
        "(Steady-State-Approximation). Vergleichen Sie mit der tatsächlichen "
        "ROIC des Unternehmens als Plausibilitätscheck."
    )

    with st.expander("ℹ️ Wie wird die implizierte ROIC berechnet?", expanded=False):
        st.markdown(r"""
Im Steady State gilt:

$$\text{NOPAT-Marge} = (\text{EBITDA\%} - \text{D\&A\%}) \times (1 - t)$$

$$\text{Reinvest-Marge} = \text{CAPEX\%} - \text{D\&A\%} + \text{NWC\%}
 \times \frac{g}{1+g}$$

$$\text{ROIC} \approx \frac{\text{NOPAT-Marge}}{\text{Reinvest-Marge}}$$

| ROIC | Einschätzung |
|------|-------------|
| > WACC | Wertschöpfung (positiver Economic Profit) |
| ≈ WACC | Grenzwertig – gerade Kapitalkosten gedeckt |
| < WACC | Wertvernichtung – Prüfen Sie die Annahmen |
""")

    seg_names = list(results.segment_implied_roic.keys())
    roic_means = [float(np.mean(results.segment_implied_roic[s])) for s in seg_names]
    roic_p5 = [float(np.percentile(results.segment_implied_roic[s], 5)) for s in seg_names]
    roic_p95 = [float(np.percentile(results.segment_implied_roic[s], 95)) for s in seg_names]

    st.plotly_chart(
        implied_roic_chart(seg_names, roic_means, roic_p5, roic_p95),
        use_container_width=True,
    )

    # Reinvestment rate detail
    if results.segment_reinvest_rates:
        st.markdown("##### Reinvestitionsquote")
        ri_cols = st.columns(len(seg_names))
        for col, name in zip(ri_cols, seg_names):
            ri = results.segment_reinvest_rates.get(name)
            if ri is not None:
                col.metric(
                    f"Reinvest – {name}",
                    f"{float(np.mean(ri)):.1%}",
                )


# ── Phase 2: Quality Score section ───────────────────────────────────────

def _render_quality_section(results) -> None:
    """Show composite valuation quality score."""
    if not results.quality_score:
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

    q = results.quality_score
    qc1, qc2 = st.columns([1, 1])
    with qc1:
        st.plotly_chart(
            quality_score_gauge(q),
            use_container_width=True,
        )
    with qc2:
        st.plotly_chart(
            quality_score_breakdown_chart(q),
            use_container_width=True,
        )
