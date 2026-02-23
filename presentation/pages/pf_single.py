"""
Portfolio Single-Asset Analysis Tab
=====================================
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from presentation.charts import (
    COLORS,
    TEMPLATE,
    cdf_with_reference,
    histogram_kde,
)


def render_single(tab) -> None:
    """Render Tab 2 (Einzeltitel-Analyse)."""
    with tab:
        if st.session_state.pf_results is None:
            st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
            return

        pf = st.session_state.pf_results
        st.header("🔍 Einzeltitel-Analyse")

        with st.expander("ℹ️ Erklärung der Kennzahlen", expanded=False):
            st.markdown(r"""
| Kennzahl | Formel | Interpretation |
|---|---|---|
| **E[Rendite]** | $\frac{E[\text{FV}]}{\text{Kurs}} - 1$ | Erwartete Rendite bei Kauf zum aktuellen Kurs |
| **P(Gewinn)** | $P(\text{FV} > \text{Kurs})$ | Wahrscheinlichkeit, dass der Fair Value über dem Kurs liegt |
| **Margin of Safety** | $\frac{\text{Median FV} - \text{Kurs}}{\text{Median FV}}$ | Sicherheitspuffer nach Benjamin Graham |
| **Kelly f*** | $\frac{E[R]}{\text{Var}(R)}$ | Optimaler Portfolioanteil (Kelly-Kriterium) |
| **VaR (5%)** | 5. Perzentil der Rendite | Maximaler Verlust in 95 % der Szenarien |
| **CVaR / ES** | $E[R \mid R \leq \text{VaR}]$ | Erwarteter Verlust im schlimmsten 5 %-Tail |
| **Sortino** | $\frac{E[R]}{\sigma_{\text{downside}}}$ | Rendite/Risiko nur mit Downside-Volatilität |
| **Omega** | $\frac{E[\max(R,0)]}{E[\max(-R,0)]}$ | Verhältnis Gewinn- zu Verlustpotenzial (> 1 = positiv) |

### Bewertungs-Ampel

| 🟢 Kaufen | 🟡 Halten | 🔴 Meiden |
|---|---|---|
| MoS > 20 % & P(Gewinn) > 65 % | MoS > 0 % & P(Gewinn) > 50 % | Sonst |
""")

        # ── Summary table ─────────────────────────────────────────────
        st.subheader("📋 Übersichtstabelle")

        summary_rows = []
        for am in pf["asset_metrics"]:
            summary_rows.append({
                "Asset": am.name,
                "Sektor": am.sector,
                "Kurs": f"{am.current_price:,.2f} €",
                "Ø Fair Value": f"{am.mean_fv:,.2f} €",
                "E[Rendite]": f"{am.expected_return:+.1%}",
                "P(Gewinn)": f"{am.prob_profit:.1%}",
                "MoS": f"{am.margin_of_safety:+.1%}",
                "Kelly f*": f"{am.kelly_fraction:.1%}",
                "VaR(5%)": f"{am.var_5:+.1%}",
                "CVaR(5%)": f"{am.cvar_5:+.1%}",
                "Sortino": f"{am.sortino_ratio:.2f}",
                "Omega": f"{am.omega_ratio:.2f}",
                "Signal": am.signal,
            })

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True, hide_index=True,
        )

        # ── Per-asset detail ──────────────────────────────────────────
        st.divider()
        st.subheader("📈 Detail-Analyse je Aktie")

        asset_tabs = st.tabs([am.name for am in pf["asset_metrics"]])

        for at, am in zip(asset_tabs, pf["asset_metrics"]):
            with at:
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("E[Rendite]", f"{am.expected_return:+.1%}")
                m2.metric("P(Gewinn)", f"{am.prob_profit:.1%}")
                m3.metric("MoS", f"{am.margin_of_safety:+.1%}")
                m4.metric("Half Kelly", f"{am.half_kelly:.1%}")
                m5.metric("Sortino", f"{am.sortino_ratio:.2f}")
                m6.metric("Omega", f"{am.omega_ratio:.2f}")

                m7, m8, m9, m10, m11, _ = st.columns(6)
                m7.metric("FV (P5)", f"{am.fv_p5:,.2f} €")
                m8.metric("FV (P50)", f"{am.median_fv:,.2f} €")
                m9.metric("FV (P95)", f"{am.fv_p95:,.2f} €")
                m10.metric("VaR (5%)", f"{am.var_5:+.1%}")
                m11.metric("CVaR (5%)", f"{am.cvar_5:+.1%}")

                st.divider()

                ch1, ch2 = st.columns(2)
                with ch1:
                    vlines = {
                        f"Kurs: {am.current_price:,.2f}": (
                            am.current_price, COLORS["negative"], "dash",
                        ),
                        f"Ø FV: {am.mean_fv:,.2f}": (
                            am.mean_fv, COLORS["positive"], "dot",
                        ),
                    }
                    st.plotly_chart(
                        histogram_kde(
                            am.fv_samples,
                            f"Fair-Value-Verteilung – {am.name}",
                            "Fair Value (€)",
                            color=COLORS["primary"],
                            vlines=vlines,
                            show_percentile_lines=False,
                        ),
                        use_container_width=True,
                    )

                with ch2:
                    st.plotly_chart(
                        cdf_with_reference(
                            am.fv_samples,
                            f"CDF – Fair Value – {am.name}",
                            "Fair Value (€)",
                            ref_value=am.current_price,
                            ref_label="Kurs",
                        ),
                        use_container_width=True,
                    )

                ch3, ch4 = st.columns(2)
                with ch3:
                    returns_pct = am.returns * 100
                    st.plotly_chart(
                        histogram_kde(
                            returns_pct,
                            f"Renditeverteilung – {am.name}",
                            "Rendite (%)",
                            color=COLORS["secondary"],
                            vlines={"Breakeven": (0, COLORS["neutral"], "solid")},
                            show_percentile_lines=False,
                        ),
                        use_container_width=True,
                    )

                with ch4:
                    fig_ud = go.Figure()
                    scenarios = [
                        ("Downside (P25)", am.downside_p25, COLORS["negative"]),
                        ("Erwartung (Ø)", am.expected_return, COLORS["primary"]),
                        ("Upside (P75)", am.upside_p75, COLORS["positive"]),
                    ]
                    fig_ud.add_trace(go.Bar(
                        x=[s[0] for s in scenarios],
                        y=[s[1] * 100 for s in scenarios],
                        marker_color=[s[2] for s in scenarios],
                        text=[f"{s[1]:+.1%}" for s in scenarios],
                        textposition="outside",
                    ))
                    fig_ud.update_layout(
                        title=f"Rendite-Szenarien – {am.name}",
                        yaxis_title="Rendite (%)",
                        template=TEMPLATE, height=440,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ud, use_container_width=True)

                with st.expander(f"💡 Interpretation – {am.name}"):
                    if am.margin_of_safety > 0.25:
                        st.success(
                            f"**Hohe Margin of Safety ({am.margin_of_safety:+.1%}):** "
                            f"Der Markt bepreist {am.name} deutlich unter dem geschätzten "
                            f"Median-Fair-Value ({am.median_fv:,.2f} € vs. {am.current_price:,.2f} €)."
                        )
                    elif am.margin_of_safety > 0:
                        st.info(
                            f"**Moderate Margin of Safety ({am.margin_of_safety:+.1%}):** "
                            f"Sicherheitspuffer vorhanden, aber begrenzt."
                        )
                    else:
                        st.warning(
                            f"**Negative Margin of Safety ({am.margin_of_safety:+.1%}):** "
                            f"Der Kurs liegt über dem Median-Fair-Value."
                        )

                    if am.omega_ratio > 1.5:
                        st.markdown(
                            f"📊 **Omega Ratio: {am.omega_ratio:.2f}** – "
                            f"Gewinnpotenzial überwiegt das Verlustrisiko deutlich."
                        )
                    elif am.omega_ratio < 1.0:
                        st.markdown(
                            f"⚠️ **Omega Ratio: {am.omega_ratio:.2f}** – "
                            f"Verlustrisiko überwiegt das Gewinnpotenzial."
                        )

                    st.markdown(
                        f"🎯 **Kelly-Empfehlung:** Optimaler Anteil = {am.kelly_fraction:.1%} "
                        f"(Half Kelly = {am.half_kelly:.1%}). "
                        + (
                            "Aggressive Allokation – Half Kelly empfohlen."
                            if am.kelly_fraction > 0.25 else
                            "Moderate Allokation."
                            if am.kelly_fraction > 0.05 else
                            "Geringer Kelly-Anteil – nur als Beimischung sinnvoll."
                        )
                    )
