"""
DCF Results Tab – Interactive charts, statistics & Excel export
================================================================
"""
from __future__ import annotations

import streamlit as st
from presentation.pages.dcf_sections import (
    render_conditional_sensitivity_section,
    render_descriptive_stats_section,
    render_distribution_section,
    render_economic_profit_section,
    render_excel_export_section,
    render_key_metrics_section,
    render_margin_of_safety_section,
    render_quality_section,
    render_roic_section,
    render_sensitivity_section,
    render_tail_risk_section,
    render_tv_ev_section,
)
from presentation.ui_helpers import render_info_interpretation
from presentation.charts import (
    convergence_chart,
    histogram_kde,
    valuation_confidence_panel,
    waterfall_chart,
)


def render_results(container) -> None:
    """Render Results step.

    Reads ``st.session_state.results`` and ``st.session_state.config``.
    """
    with container:
        if st.session_state.results is None:
            st.warning(
                "⚠️ Noch keine Ergebnisse vorhanden. Wechseln Sie zu **🎲 Simulation**, starten Sie den Lauf und kehren Sie dann hierher zurück."
            )
            return

        results = st.session_state.results
        config = st.session_state.config

        st.header("📈 Simulationsergebnisse")
        render_info_interpretation()
        tab_overview, tab_risk, tab_drivers, tab_quality, tab_detail = st.tabs([
            "🧭 Überblick",
            "⚠️ Risiko",
            "🧠 Treiber",
            "✅ Qualität",
            "🧪 Detail & Export",
        ])

        with tab_overview:
            render_key_metrics_section(results)
            st.divider()

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

        with tab_risk:
            render_tail_risk_section(results)
            st.divider()
            render_margin_of_safety_section(results)
            st.divider()
            render_economic_profit_section(results)

        with tab_drivers:
            render_sensitivity_section(results)
            st.divider()
            render_conditional_sensitivity_section(results)

        with tab_quality:
            render_tv_ev_section(results)
            st.divider()
            render_roic_section(results, config)
            st.divider()
            render_quality_section(results)
            st.divider()

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

                quality_score = getattr(results, "quality_score", None)
                quality_total = quality_score.get("total") if quality_score else None
                tail_ratio = getattr(results, "equity_tail_ratio", None)
                st.plotly_chart(
                    valuation_confidence_panel(
                        ci_relative_width_pct=float(pct_width),
                        tail_ratio=tail_ratio,
                        quality_total=quality_total,
                    ),
                    use_container_width=True,
                )

        with tab_detail:
            render_descriptive_stats_section(results)
            st.divider()
            render_distribution_section(results)
            st.divider()

            with st.expander("🧪 Erweiterte Detailcharts (lazy)", expanded=False):
                enable_detail_charts = st.checkbox(
                    "Detailcharts laden",
                    value=False,
                    key="dcf_lazy_detail_charts",
                    help="Lädt schwere Detailvisualisierungen nur bei Bedarf.",
                )
                if enable_detail_charts:
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
                else:
                    st.info("Detailcharts sind aktuell deaktiviert. Aktivieren Sie **Detailcharts laden**, wenn Sie Segmentverteilungen im Detail prüfen möchten.")

            st.divider()
            render_excel_export_section(config, results)
