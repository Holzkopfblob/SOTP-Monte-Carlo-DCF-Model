"""
DCF Segments Tab – Per-segment FCFF assumptions
=================================================
"""
from __future__ import annotations

import streamlit as st

from domain.models import (
    DistributionConfig,
    RevenueGrowthMode,
    SegmentConfig,
    TerminalValueMethod,
)
from presentation.ui_helpers import (
    render_distribution_input,
    render_info_distributions,
    render_info_fade_model,
    render_info_fcff,
    render_info_terminal_value,
    render_info_wacc,
)
from presentation.charts import revenue_fade_preview, parameter_fade_preview


def render_segments(tab, n_segments: int) -> list[SegmentConfig]:
    """Render Tab 2 (Segmente) and return the list of segment configs."""
    segment_configs: list[SegmentConfig] = []

    with tab:
        st.header("Segment-Konfiguration")
        render_info_fcff()
        render_info_wacc()
        render_info_distributions()
        render_info_terminal_value()
        render_info_fade_model()

        for i in range(n_segments):
            with st.expander(f"📦 Segment {i + 1}", expanded=(i == 0)):
                # ── Basic info ────────────────────────────────────────
                sc1, sc2 = st.columns(2)
                seg_name = sc1.text_input(
                    "Segmentname", value=f"Segment {i + 1}",
                    key=f"seg_{i}_name",
                )
                base_rev = sc1.number_input(
                    "Basisumsatz (Mio. / Jahr 0)",
                    value=1_000.0, min_value=0.01,
                    key=f"seg_{i}_basrev", format="%.1f",
                )
                forecast_yrs = sc2.number_input(
                    "Detail-Prognosezeitraum (Jahre)",
                    value=5, min_value=1, max_value=30,
                    key=f"seg_{i}_fyrs",
                )

                st.markdown("---")
                st.markdown("##### 📐 Werttreiber")

                # ── Revenue growth mode ───────────────────────────────

                growth_mode_str = st.selectbox(
                    "Umsatzwachstums-Modell",
                    options=[m.value for m in RevenueGrowthMode],
                    key=f"seg_{i}_growth_mode",
                    help="Konstant: gleiche Rate jedes Jahr. "
                         "Fade: hohes Anfangswachstum konvergiert zum Terminal-Wachstum.",
                )
                growth_mode = RevenueGrowthMode(growth_mode_str)

                fade_speed_val = 0.5
                if growth_mode == RevenueGrowthMode.FADE:
                    fade_col1, _fade_col2 = st.columns(2)
                    fade_speed_val = fade_col1.slider(
                        "Fade-Geschwindigkeit (λ)",
                        min_value=0.05, max_value=2.0, value=0.5, step=0.05,
                        key=f"seg_{i}_fade_speed",
                        help="Höher = schnellere Konvergenz zum Terminal-Wachstum. "
                             "0.3 = langsam, 0.5 = mittel, 1.0+ = schnell.",
                    )
                    st.caption(
                        "💡 Das initiale Wachstum (unten) fällt exponentiell zum "
                        "TV-Wachstum ab. Die Vorschau zeigt den resultierenden Pfad."
                    )

                rev_growth = render_distribution_input(
                    "Umsatzwachstum (initial)" if growth_mode == RevenueGrowthMode.FADE else "Umsatzwachstum",
                    f"s{i}_rg", 5.0, is_percentage=True,
                    help_text="Jährliches Umsatzwachstum (initial bei Fade-Modell, konstant sonst).",
                )
                ebitda_m = render_distribution_input(
                    "EBITDA-Marge" + (" (initial)" if growth_mode == RevenueGrowthMode.FADE else ""),
                    f"s{i}_em", 20.0, is_percentage=True,
                    help_text="EBITDA als Anteil am Umsatz.",
                )
                da_pct = render_distribution_input(
                    "D&A (% Umsatz)" + (" (initial)" if growth_mode == RevenueGrowthMode.FADE else ""),
                    f"s{i}_da", 3.0, is_percentage=True,
                    help_text="Abschreibungen als Anteil am Umsatz.",
                )
                tax_r = render_distribution_input(
                    "Steuersatz" + (" (initial)" if growth_mode == RevenueGrowthMode.FADE else ""),
                    f"s{i}_tx", 25.0, is_percentage=True,
                    help_text="Effektiver Körperschaftssteuersatz.",
                )
                capex = render_distribution_input(
                    "CAPEX (% Umsatz)" + (" (initial)" if growth_mode == RevenueGrowthMode.FADE else ""),
                    f"s{i}_cx", 5.0, is_percentage=True,
                    help_text="Investitionsausgaben als Anteil am Umsatz.",
                )
                nwc = render_distribution_input(
                    "NWC (% ΔUmsatz)" + (" (initial)" if growth_mode == RevenueGrowthMode.FADE else ""),
                    f"s{i}_nwc", 10.0, is_percentage=True,
                    help_text="Working-Capital-Veränderung als Anteil der Umsatzveränderung.",
                )

                # ── Parameter fade terminals (Phase 3) ────────────────
                ebitda_m_term: DistributionConfig | None = None
                da_pct_term:   DistributionConfig | None = None
                tax_r_term:    DistributionConfig | None = None
                capex_term:    DistributionConfig | None = None
                nwc_term:      DistributionConfig | None = None

                if growth_mode == RevenueGrowthMode.FADE:
                    enable_param_fade = st.checkbox(
                        "📉 Parameter-Fade für Margen / Quoten aktivieren",
                        value=False, key=f"seg_{i}_param_fade",
                        help="Wenn aktiv, konvergieren EBITDA-Marge, D&A, Steuer, "
                             "CAPEX und NWC ebenfalls exponentiell zu einem "
                             "Terminal-Wert (gleiche Fade-Geschwindigkeit λ wie "
                             "das Umsatzwachstum).",
                    )
                    if enable_param_fade:
                        st.caption(
                            "💡 Geben Sie die langfristigen Zielwerte ein. "
                            "Die Parameter faden mit derselben Geschwindigkeit (λ) "
                            "wie das Umsatzwachstum."
                        )
                        ebitda_m_term = render_distribution_input(
                            "EBITDA-Marge (terminal)", f"s{i}_em_term", 18.0,
                            is_percentage=True,
                            help_text="Langfristige Gleichgewichts-EBITDA-Marge.",
                        )
                        da_pct_term = render_distribution_input(
                            "D&A (% Umsatz) (terminal)", f"s{i}_da_term", 3.0,
                            is_percentage=True,
                            help_text="Langfristige D&A-Quote.",
                        )
                        tax_r_term = render_distribution_input(
                            "Steuersatz (terminal)", f"s{i}_tx_term", 25.0,
                            is_percentage=True,
                            help_text="Langfristiger effektiver Steuersatz.",
                        )
                        capex_term = render_distribution_input(
                            "CAPEX (% Umsatz) (terminal)", f"s{i}_cx_term", 4.0,
                            is_percentage=True,
                            help_text="Langfristige CAPEX-Quote (Erhaltungsinvestitionen).",
                        )
                        nwc_term = render_distribution_input(
                            "NWC (% ΔUmsatz) (terminal)", f"s{i}_nwc_term", 8.0,
                            is_percentage=True,
                            help_text="Langfristige Working-Capital-Intensität.",
                        )
                wacc_d = render_distribution_input(
                    "WACC", f"s{i}_wacc", 9.0, is_percentage=True,
                    help_text="Gewichteter Kapitalkostensatz für dieses Segment.",
                )

                # ── Terminal value ────────────────────────────────────
                st.markdown("---")
                st.markdown("##### 🏁 Terminal Value")

                tv_method_str = st.selectbox(
                    "Methode",
                    options=[m.value for m in TerminalValueMethod],
                    key=f"seg_{i}_tv_method",
                )
                tv_method = TerminalValueMethod(tv_method_str)

                if tv_method == TerminalValueMethod.GORDON_GROWTH:
                    tv_growth = render_distribution_input(
                        "TV-Wachstumsrate", f"s{i}_tvg", 2.0,
                        is_percentage=True,
                        help_text="Ewige Wachstumsrate g (muss < WACC sein).",
                    )
                    tv_multiple = DistributionConfig(fixed_value=10.0)
                else:
                    tv_growth = DistributionConfig(fixed_value=0.02)
                    tv_multiple = render_distribution_input(
                        "Exit-Multiple (EV/EBITDA)", f"s{i}_evm", 10.0,
                        help_text="EV/EBITDA-Multiple im Endjahr.",
                    )

                # ── Fade-Modell Vorschau ──────────────────────────────
                if growth_mode == RevenueGrowthMode.FADE:
                    _g_init = (
                        rev_growth.fixed_value
                        if rev_growth.dist_type.value == "Fest (Deterministisch)"
                        else rev_growth.mean
                    )
                    _g_term = (
                        tv_growth.fixed_value
                        if tv_growth.dist_type.value == "Fest (Deterministisch)"
                        else tv_growth.mean
                    )
                    st.plotly_chart(
                        revenue_fade_preview(
                            g_initial=_g_init,
                            g_terminal=_g_term,
                            fade_speed=fade_speed_val,
                            forecast_years=int(forecast_yrs),
                        ),
                        use_container_width=True,
                    )

                    # ── Parameter-Fade Vorschau ───────────────────────
                    if ebitda_m_term is not None:
                        def _pval(cfg: DistributionConfig) -> float:
                            if cfg.dist_type.value == "Fest (Deterministisch)":
                                return cfg.fixed_value * 100
                            return cfg.mean * 100

                        fade_params: dict[str, tuple[float, float]] = {}
                        for lbl, init_c, term_c in [
                            ("EBITDA-Marge", ebitda_m, ebitda_m_term),
                            ("D&A (% Umsatz)", da_pct, da_pct_term),
                            ("Steuersatz", tax_r, tax_r_term),
                            ("CAPEX (% Umsatz)", capex, capex_term),
                            ("NWC (% ΔUmsatz)", nwc, nwc_term),
                        ]:
                            if term_c is not None:
                                fade_params[lbl] = (_pval(init_c), _pval(term_c))

                        if fade_params:
                            st.plotly_chart(
                                parameter_fade_preview(
                                    fade_speed=fade_speed_val,
                                    forecast_years=int(forecast_yrs),
                                    params=fade_params,
                                ),
                                use_container_width=True,
                            )

                # ── Build config object ───────────────────────────────
                segment_configs.append(SegmentConfig(
                    name=seg_name,
                    base_revenue=float(base_rev),
                    forecast_years=int(forecast_yrs),
                    revenue_growth=rev_growth,
                    ebitda_margin=ebitda_m,
                    da_pct_revenue=da_pct,
                    tax_rate=tax_r,
                    capex_pct_revenue=capex,
                    nwc_pct_delta_revenue=nwc,
                    wacc=wacc_d,
                    terminal_method=tv_method,
                    terminal_growth_rate=tv_growth,
                    exit_multiple=tv_multiple,
                    revenue_growth_mode=growth_mode,
                    fade_speed=fade_speed_val,
                    # Phase 3: parameter fade terminals
                    ebitda_margin_terminal=ebitda_m_term,
                    da_pct_revenue_terminal=da_pct_term,
                    tax_rate_terminal=tax_r_term,
                    capex_pct_revenue_terminal=capex_term,
                    nwc_pct_delta_revenue_terminal=nwc_term,
                ))

    return segment_configs
