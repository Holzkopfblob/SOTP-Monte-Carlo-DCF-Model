"""
SOTP Monte-Carlo DCF Simulation – Streamlit Application
========================================================

Main entry point.  Run with:

    streamlit run app.py

The application uses four tabs:
    1. Setup          – Global simulation parameters & corporate bridge
    2. Segmente       – Per-segment FCFF assumptions (deterministic or stochastic)
    3. Simulation     – Launch the Monte-Carlo engine
    4. Ergebnisse     – Interactive charts, statistics & Excel export
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import skew, kurtosis

from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    TerminalValueMethod,
)
from application.simulation_service import SimulationService
from infrastructure.excel_export import ExcelExporter
from presentation.ui_helpers import (
    render_distribution_input,
    render_info_corporate_bridge,
    render_info_distributions,
    render_info_fcff,
    render_info_interpretation,
    render_info_monte_carlo,
    render_info_sotp,
    render_info_terminal_value,
    render_info_wacc,
    render_info_fade_model,
)
from presentation.charts import (
    cdf_plot,
    convergence_chart,
    histogram_kde,
    price_histogram,
    revenue_fade_preview,
    tornado_chart,
    waterfall_chart,
)


# ══════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SOTP Monte-Carlo DCF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    /* Tab gap */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 500;
    }
    /* Expander headers */
    details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════

_SETUP_KEYS = [
    "setup_n_sim", "setup_seed", "setup_n_seg",
    "setup_corp_costs", "setup_corp_disc", "setup_net_debt", "setup_shares",
]
_DIST_PARAMS = ["rg", "em", "da", "tx", "cx", "nwc", "wacc", "tvg", "evm"]
_DIST_SUFFIXES = [
    "_dtype", "_fixed",
    "_n_mu", "_n_sig",
    "_ln_mu", "_ln_sig",
    "_tri_lo", "_tri_mo", "_tri_hi",
    "_uni_lo", "_uni_hi",
    "_pert_lo", "_pert_mo", "_pert_hi",
]


def _collect_config() -> dict:
    """Collect current widget values into a serialisable dict."""
    cfg: dict = {"version": 1, "saved_at": datetime.now().isoformat()}

    # Setup parameters
    setup: dict = {}
    for k in _SETUP_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            setup[k] = int(v) if isinstance(v, (int, np.integer)) else float(v)
    cfg["setup"] = setup

    # Per-segment parameters
    n_seg = int(st.session_state.get("setup_n_seg", 1))
    segments: list[dict] = []
    for i in range(n_seg):
        seg: dict = {}
        for suffix in ["_name", "_basrev", "_fyrs", "_tv_method"]:
            key = f"seg_{i}{suffix}"
            if key in st.session_state:
                v = st.session_state[key]
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                seg[key] = v
        for param in _DIST_PARAMS:
            prefix = f"s{i}_{param}"
            for sfx in _DIST_SUFFIXES:
                key = f"{prefix}{sfx}"
                if key in st.session_state:
                    v = st.session_state[key]
                    if isinstance(v, (np.integer,)):
                        v = int(v)
                    elif isinstance(v, (np.floating,)):
                        v = float(v)
                    seg[key] = v
        segments.append(seg)
    cfg["segments"] = segments
    return cfg


def _apply_config(cfg: dict) -> None:
    """Write config dict into session_state and rerun."""
    import re

    # Clear stale segment keys so old segments don't persist
    seg_pattern = re.compile(r"^(seg_\d+_|s\d+_)")
    for k in list(st.session_state.keys()):
        if seg_pattern.match(k):
            del st.session_state[k]

    # Apply setup keys
    for k, v in cfg.get("setup", {}).items():
        st.session_state[k] = v

    # Apply segment keys
    for seg_data in cfg.get("segments", []):
        for k, v in seg_data.items():
            st.session_state[k] = v

    st.session_state["_config_just_loaded"] = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Title & sidebar
# ══════════════════════════════════════════════════════════════════════════

st.title("📊 Sum-of-the-Parts Monte-Carlo DCF Modell")
st.caption(
    "Stochastische Unternehmensbewertung · FCFF-Ansatz · "
    "Vektorisierte Simulation"
)

with st.sidebar:
    st.header("SOTP MC-DCF")
    st.markdown("---")
    if "results" in st.session_state and st.session_state.results is not None:
        r = st.session_state.results
        st.success("Simulation abgeschlossen")
        st.metric("Ø Enterprise Value", f"{np.mean(r.total_ev):,.1f}")
        st.metric("Ø Equity Value", f"{np.mean(r.equity_values):,.1f}")
        st.metric(
            "Ø Preis/Aktie",
            f"{np.mean(r.price_per_share):,.2f}",
        )
        st.metric("Iterationen", f"{r.n_simulations:,}")
    else:
        st.info("Noch keine Simulation durchgeführt.")

    # ── Save / Load configuration ─────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Speichern / Laden")

    save_name = st.text_input(
        "Modellname",
        value="SOTP_Modell",
        key="save_model_name",
        help="Dateiname für den JSON-Export.",
    )
    cfg_json = json.dumps(_collect_config(), indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Konfiguration speichern",
        data=cfg_json,
        file_name=f"{save_name}_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("")
    uploaded_cfg = st.file_uploader(
        "Konfiguration laden (.json)",
        type=["json"],
        key="config_upload",
    )
    if uploaded_cfg is not None:
        if st.button(
            "⬆️ Konfiguration anwenden",
            use_container_width=True,
            type="primary",
        ):
            try:
                raw = uploaded_cfg.getvalue().decode("utf-8")
                loaded = json.loads(raw)
                if "setup" not in loaded:
                    st.error("Ungültiges Dateiformat – 'setup' fehlt.")
                else:
                    _apply_config(loaded)  # calls st.rerun()
            except json.JSONDecodeError:
                st.error("Ungültige JSON-Datei.")
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")

    if st.session_state.pop("_config_just_loaded", False):
        st.success("✅ Konfiguration erfolgreich geladen!")

    st.markdown("---")
    st.caption("Built with Streamlit · NumPy · Plotly")


# ══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "config" not in st.session_state:
    st.session_state.config = None


# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

tab_setup, tab_segments, tab_sim, tab_results = st.tabs([
    "⚙️ Setup",
    "🏢 Segmente",
    "🎲 Simulation",
    "📈 Ergebnisse",
])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 – SETUP
# ──────────────────────────────────────────────────────────────────────────

with tab_setup:
    st.header("Modell-Konfiguration")
    render_info_sotp()
    render_info_monte_carlo()

    col_a, col_b = st.columns(2)
    with col_a:
        n_simulations = st.number_input(
            "Anzahl Monte-Carlo-Iterationen",
            min_value=1_000,
            max_value=500_000,
            value=10_000,
            step=1_000,
            help="Mehr Iterationen → genauere Ergebnisse, längere Laufzeit.",
            key="setup_n_sim",
        )
        random_seed = st.number_input(
            "Random Seed (Reproduzierbarkeit)",
            value=42,
            min_value=0,
            key="setup_seed",
        )
    with col_b:
        n_segments = st.number_input(
            "Anzahl Geschäftssegmente",
            min_value=1,
            max_value=20,
            value=2,
            help="Für jedes Segment wird ein separater DCF berechnet.",
            key="setup_n_seg",
        )

    st.divider()

    # ── Corporate bridge ──────────────────────────────────────────────
    st.subheader("🏛️ Unternehmensbrücke (Corporate Bridge)")
    render_info_corporate_bridge()

    c1, c2, c3, c4 = st.columns(4)
    annual_corp_costs = c1.number_input(
        "Jährl. Holdingkosten (Mio.)",
        value=50.0, min_value=0.0, format="%.1f",
        help="Laufende Kosten der Holding-Gesellschaft p.a.",
        key="setup_corp_costs",
    )
    corp_discount = c2.number_input(
        "Diskontierung Holding (%)",
        value=9.0, min_value=0.1, format="%.2f",
        help="Diskontierungssatz für die Perpetuity der Holdingkosten.",
        key="setup_corp_disc",
    )
    net_debt = c3.number_input(
        "Nettoverschuldung (Mio.)",
        value=500.0, format="%.1f",
        help="Finanzschulden − Cash & Äquivalente.",
        key="setup_net_debt",
    )
    shares = c4.number_input(
        "Aktien ausstehend (Mio.)",
        value=100.0, min_value=0.01, format="%.2f",
        help="Voll verwässerte Aktienanzahl.",
        key="setup_shares",
    )

    # ── Stochastic Corporate Bridge ───────────────────────────────────
    st.markdown("")
    enable_stoch_bridge = st.checkbox(
        "🎲 Stochastische Corporate Bridge aktivieren",
        value=False,
        key="setup_stoch_bridge",
        help="Modelliert Holdingkosten, Nettoverschuldung und Aktienanzahl "
             "als Wahrscheinlichkeitsverteilungen statt fester Werte.",
    )

    stoch_corp_costs = None
    stoch_net_debt = None
    stoch_shares = None

    if enable_stoch_bridge:
        st.caption(
            "Definieren Sie Verteilungen für die Corporate-Bridge-Parameter. "
            "Die oben eingegebenen Festwerte werden als Default-Mittelwerte verwendet."
        )
        with st.expander("📐 Stochastische Holdingkosten", expanded=True):
            stoch_corp_costs = render_distribution_input(
                "Holdingkosten (Mio. p.a.)", "bridge_cc",
                default_value=annual_corp_costs,
                is_percentage=False,
                help_text="Jährliche Holdingkosten als Verteilung.",
            )
        with st.expander("📐 Stochastische Nettoverschuldung", expanded=True):
            stoch_net_debt = render_distribution_input(
                "Nettoverschuldung (Mio.)", "bridge_nd",
                default_value=net_debt,
                is_percentage=False,
                help_text="Nettoverschuldung als Verteilung (z.B. bei geplanten Tilgungen / Akquisitionen).",
            )
        with st.expander("📐 Stochastische Aktienanzahl", expanded=True):
            stoch_shares = render_distribution_input(
                "Aktien ausstehend (Mio.)", "bridge_sh",
                default_value=shares,
                is_percentage=False,
                help_text="Verwässerte Aktienanzahl als Verteilung (z.B. bei Rückkaufprogrammen / Optionen).",
            )


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 – SEGMENTE
# ──────────────────────────────────────────────────────────────────────────

segment_configs: list[SegmentConfig] = []

with tab_segments:
    st.header("Segment-Konfiguration")
    render_info_fcff()
    render_info_wacc()
    render_info_distributions()
    render_info_terminal_value()

    for i in range(int(n_segments)):
        with st.expander(f"📦 Segment {i + 1}", expanded=(i == 0)):
            # ── Basic info ────────────────────────────────────────────
            sc1, sc2 = st.columns(2)
            seg_name = sc1.text_input(
                "Segmentname",
                value=f"Segment {i + 1}",
                key=f"seg_{i}_name",
            )
            base_rev = sc1.number_input(
                "Basisumsatz (Mio. / Jahr 0)",
                value=1_000.0,
                min_value=0.01,
                key=f"seg_{i}_basrev",
                format="%.1f",
            )
            forecast_yrs = sc2.number_input(
                "Detail-Prognosezeitraum (Jahre)",
                value=5,
                min_value=1,
                max_value=30,
                key=f"seg_{i}_fyrs",
            )

            st.markdown("---")
            st.markdown("##### 📐 Werttreiber")

            # ── Revenue growth mode ───────────────────────────────────
            render_info_fade_model()

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
                fade_col1, fade_col2 = st.columns(2)
                fade_speed_val = fade_col1.slider(
                    "Fade-Geschwindigkeit (λ)",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
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
                "EBITDA-Marge", f"s{i}_em", 20.0, is_percentage=True,
                help_text="EBITDA als Anteil am Umsatz.",
            )
            da_pct = render_distribution_input(
                "D&A (% Umsatz)", f"s{i}_da", 3.0, is_percentage=True,
                help_text="Abschreibungen als Anteil am Umsatz.",
            )
            tax_r = render_distribution_input(
                "Steuersatz", f"s{i}_tx", 25.0, is_percentage=True,
                help_text="Effektiver Körperschaftssteuersatz.",
            )
            capex = render_distribution_input(
                "CAPEX (% Umsatz)", f"s{i}_cx", 5.0, is_percentage=True,
                help_text="Investitionsausgaben als Anteil am Umsatz.",
            )
            nwc = render_distribution_input(
                "NWC (% ΔUmsatz)", f"s{i}_nwc", 10.0, is_percentage=True,
                help_text="Working-Capital-Veränderung als Anteil der Umsatzveränderung.",
            )
            wacc_d = render_distribution_input(
                "WACC", f"s{i}_wacc", 9.0, is_percentage=True,
                help_text="Gewichteter Kapitalkostensatz für dieses Segment.",
            )

            # ── Terminal value ────────────────────────────────────────
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
                    "TV-Wachstumsrate", f"s{i}_tvg", 2.0, is_percentage=True,
                    help_text="Ewige Wachstumsrate g (muss < WACC sein).",
                )
                tv_multiple = DistributionConfig(fixed_value=10.0)
            else:
                tv_growth = DistributionConfig(fixed_value=0.02)
                tv_multiple = render_distribution_input(
                    "Exit-Multiple (EV/EBITDA)", f"s{i}_evm", 10.0,
                    help_text="EV/EBITDA-Multiple im Endjahr.",
                )

            # ── Fade-Modell Vorschau ──────────────────────────────────
            if growth_mode == RevenueGrowthMode.FADE:
                # Determine initial g and terminal g for preview
                # Use the fixed_value / mean as best guess for preview
                _g_init = rev_growth.fixed_value if rev_growth.dist_type.value == "Fest (Deterministisch)" else rev_growth.mean
                _g_term = tv_growth.fixed_value if tv_growth.dist_type.value == "Fest (Deterministisch)" else tv_growth.mean
                st.plotly_chart(
                    revenue_fade_preview(
                        g_initial=_g_init,
                        g_terminal=_g_term,
                        fade_speed=fade_speed_val,
                        forecast_years=int(forecast_yrs),
                    ),
                    use_container_width=True,
                )

            # ── Build config object ───────────────────────────────────
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
            ))


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 – SIMULATION
# ──────────────────────────────────────────────────────────────────────────

with tab_sim:
    st.header("Monte-Carlo-Simulation starten")

    # Summary of configuration
    st.info(
        f"**Konfiguration:** {int(n_segments)} Segment(e) · "
        f"{int(n_simulations):,} Iterationen · Seed {int(random_seed)}"
    )

    if segment_configs:
        with st.expander("Segment-Übersicht", expanded=True):
            overview_rows = []
            for sc in segment_configs:
                overview_rows.append({
                    "Segment": sc.name,
                    "Basisumsatz": f"{sc.base_revenue:,.1f} Mio.",
                    "Prognosejahre": sc.forecast_years,
                    "TV-Methode": sc.terminal_method.value,
                })
            st.table(pd.DataFrame(overview_rows))

    st.markdown("")
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_button = st.button(
            "🚀 Simulation starten",
            type="primary",
            use_container_width=True,
        )

    if run_button:
        if not segment_configs:
            st.error("Bitte konfigurieren Sie mindestens ein Segment.")
        else:
            config = SimulationConfig(
                n_simulations=int(n_simulations),
                random_seed=int(random_seed),
                segments=segment_configs,
                corporate_bridge=CorporateBridgeConfig(
                    annual_corporate_costs=float(annual_corp_costs),
                    corporate_cost_discount_rate=float(corp_discount) / 100.0,
                    net_debt=float(net_debt),
                    shares_outstanding=float(shares),
                    stochastic_corporate_costs=stoch_corp_costs,
                    stochastic_net_debt=stoch_net_debt,
                    stochastic_shares=stoch_shares,
                ),
            )

            progress_bar = st.progress(0, text="Initialisiere Simulation …")
            progress_bar.progress(10, text="Generiere stochastische Samples …")

            results = SimulationService.run_simulation(config)

            progress_bar.progress(90, text="Berechne Statistiken …")
            st.session_state.results = results
            st.session_state.config = config
            progress_bar.progress(100, text="Fertig!")

            st.success(
                f"✅ Simulation abgeschlossen – "
                f"{int(n_simulations):,} Szenarien berechnet."
            )
            st.balloons()

    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown(
            "➡️ Wechseln Sie zum Tab **📈 Ergebnisse** "
            "für die vollständige Auswertung."
        )


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 – ERGEBNISSE
# ──────────────────────────────────────────────────────────────────────────

with tab_results:
    if st.session_state.results is None:
        st.warning(
            "⚠️ Bitte führen Sie zunächst eine Simulation durch "
            "(Tab **🎲 Simulation**)."
        )
    else:
        results = st.session_state.results
        config  = st.session_state.config

        st.header("📈 Simulationsergebnisse")
        render_info_interpretation()

        # ── Key metrics row ───────────────────────────────────────────
        ev_stats = SimulationService.compute_statistics(results.total_ev)
        eq_stats = SimulationService.compute_statistics(results.equity_values)
        ps_stats = SimulationService.compute_statistics(results.price_per_share)

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
            stats_data[f"EV – {seg_name}"] = (
                SimulationService.compute_statistics(seg_ev)
            )

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
                price_histogram(results.price_per_share),
                use_container_width=True,
            )

        # Additional CDF for price per share
        st.plotly_chart(
            cdf_plot(
                results.price_per_share,
                "CDF – Preis je Aktie",
                "Preis / Aktie",
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
            ),
            use_container_width=True,
        )

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

            # Convergence assessment
            final_width = results.convergence_ci_high[-1] - results.convergence_ci_low[-1]
            final_mean = results.convergence_means[-1]
            pct_width = (final_width / abs(final_mean) * 100) if abs(final_mean) > 0 else 0

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

        # ── KDE / Verteilungsparameter für Portfolio-App ──────────────
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

        # Prominent parameter display
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

        # Recommendation
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

        # Copyable text block
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
