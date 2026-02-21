"""
Portfolio-Strukturierung & Optimierung – Streamlit Application
===============================================================

Unabhängige App für die statistische Analyse und optimale Gewichtung
eines Aktienportfolios basierend auf Fair-Value-Schätzungen.

Starten mit::

    streamlit run portfolio_app.py --server.port 8502

Architektur
-----------
- **Berechnung** → ``application.portfolio_service``  (PortfolioAnalyser)
- **Charts**     → ``presentation.charts``
- **UI**         → dieses Modul (reine Streamlit-Widgets & Layout)

Features
--------
1. Eingabe beliebig vieler Bewertungen (Fair Value, Kurs, Verteilungstyp)
2. Einzeltitel-Analyse (MoS, Kelly, P(Gewinn), VaR, CVaR, Sortino, Omega)
3. Sieben Optimierungsmethoden (Max Sharpe, Min Vol, Risk Parity,
   Min CVaR, Max Diversifikation, Multi-Asset Kelly, Gleichgewicht)
4. Efficient Frontier mit CML
5. Korrelationsmatrix (Cluster-basiert, manuell, unkorreliert)
6. Stress-Tests (Marktschock, Korrelationsstress, Sektorkrisen)
7. Eingebettete Erklärungen mit LaTeX-Formeln
"""
from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from application.portfolio_service import (
    AssetInput,
    PortfolioAnalyser,
    PortfolioResult,
    generate_fv_samples,
)
from presentation.charts import (
    COLORS,
    TEMPLATE,
    PALETTE_EXTENDED,
    cdf_with_reference,
    correlation_heatmap,
    histogram_kde,
    portfolio_weights_comparison,
    stress_comparison_chart,
)


# ══════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio-Optimierung",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #9467bd;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 500; }
    details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

DIST_OPTIONS = [
    "Aus DCF-App (μ, σ, Schiefe)",
    "Normal",
    "Lognormal",
    "PERT",
    "Dreiecksverteilung",
    "Gleichverteilung",
]

SECTOR_LIST = [
    "Technologie", "Gesundheit", "Finanzen", "Industrie",
    "Konsumgüter", "Energie", "Immobilien", "Versorger",
    "Telekommunikation", "Grundstoffe", "Sonstige",
]

# Display order for optimisation methods
METHOD_ORDER = [
    "Gleichgewicht (1/N)",
    "Max Sharpe",
    "Min Volatilität",
    "Risk Parity",
    "Min CVaR",
    "Max Diversifikation",
    "Kelly (Multi-Asset)",
]


# ══════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════

st.title("💼 Portfolio-Strukturierung & Optimierung")
st.caption(
    "Statistische Analyse · Markowitz · Kelly · Risk Parity · "
    "Min CVaR · Max Diversifikation · Monte-Carlo Fair-Value-Verteilungen"
)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("💼 Portfolio-Tool")
    st.markdown("---")
    st.markdown(
        "Geben Sie für jede Aktie Ihre Bewertung ein und erhalten Sie "
        "eine optimale Portfoliostruktur mit **7 Methoden** im Vergleich."
    )
    st.markdown("---")

    st.subheader("⚙️ Globale Einstellungen")
    n_mc_sim = st.number_input(
        "MC-Simulationen je Asset",
        min_value=5_000, max_value=200_000, value=50_000, step=5_000,
        help="Mehr Samples → genauere Verteilungen, aber längere Berechnung.",
    )
    global_seed = st.number_input("Random Seed", value=42, min_value=0)
    risk_free_pct = st.number_input(
        "Risikofreier Zins (%)", value=3.0, min_value=0.0, max_value=20.0,
        format="%.2f",
        help="z.B. 10J-Bundesanleihe oder US Treasury Yield.",
    )

    st.markdown("---")

    # ── Save / Load ───────────────────────────────────────────────────
    st.subheader("💾 Konfiguration")
    st.caption("Portfolio-Setup als JSON speichern oder laden.")
    uploaded = st.file_uploader("JSON laden", type=["json"], key="pf_upload")

    st.markdown("---")
    st.caption("Built with Streamlit · NumPy · SciPy · Plotly")


# ══════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════

if "pf_results" not in st.session_state:
    st.session_state.pf_results = None


# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

tab_input, tab_single, tab_portfolio, tab_frontier, tab_stress = st.tabs([
    "📝 Bewertungen eingeben",
    "🔍 Einzeltitel-Analyse",
    "📊 Portfolio-Optimierung",
    "📈 Efficient Frontier",
    "⚡ Stress-Tests",
])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 – BEWERTUNGEN EINGEBEN
# ──────────────────────────────────────────────────────────────────────────

with tab_input:
    st.header("📝 Bewertungen & Preise eingeben")

    with st.expander("ℹ️ Anleitung – So nutzen Sie dieses Tool", expanded=False):
        st.markdown("""
### Workflow

1. **Anzahl Aktien festlegen** und für jede Aktie die Bewertungsparameter eingeben
2. **Fair-Value-Verteilung wählen**: Wie sicher sind Sie sich bei Ihrer Bewertung?
   - **🔗 Aus DCF-App (μ, σ, Schiefe)**: Direkte Übernahme der Parameter aus der SOTP-DCF-Simulation **(empfohlen!)**
   - **Normal**: Symmetrische Unsicherheit um den geschätzten Fair Value
   - **Lognormal**: Rechtsschief – positives Upside, begrenztes Downside (typisch für Aktien)
   - **PERT**: Experten-Dreipunktschätzung (Min / Wahrscheinlichster / Max)
   - **Dreiecksverteilung**: Einfache Min / Mode / Max-Schätzung
   - **Gleichverteilung**: Maximale Unsicherheit in einem Intervall
3. **Aktueller Börsenkurs**: Der Marktkurs, zu dem Sie kaufen würden
4. **Analyse starten**: Das Tool berechnet für jede Aktie und das Gesamtportfolio
   statistische Kennzahlen und optimale Gewichtungen

### Welche Verteilung wählen?

| Situation | Empfohlene Verteilung |
|---|---|
| Sie haben die DCF-App genutzt | **Aus DCF-App (μ, σ, Schiefe)** |
| Sie haben ein DCF-Modell mit klarem Ergebnis (z.B. Fair Value ≈ 85 €) | **Normal** (μ = 85, σ = 10-15) |
| Aktie hat mehr Upside-Potential als Downside-Risiko | **Lognormal** |
| Sie haben Best/Base/Worst-Case geschätzt | **PERT** oder **Dreieck** |
| Sie haben nur eine grobe Range (z.B. 60–100 €) | **Gleichverteilung** |
""")

    # Apply loaded JSON configuration
    if uploaded is not None:
        try:
            loaded_cfg = json.loads(uploaded.read().decode("utf-8"))
            st.session_state["_pf_loaded_cfg"] = loaded_cfg
            st.success("✅ Konfiguration geladen.")
        except Exception as exc:
            st.error(f"Fehler beim Laden: {exc}")

    n_assets = st.number_input(
        "Anzahl Aktien / Assets", min_value=1, max_value=25, value=3,
        help="Geben Sie für jede Aktie Ihre Bewertungsparameter ein.",
    )

    st.divider()

    asset_configs: list[dict] = []

    for i in range(int(n_assets)):
        with st.expander(f"📌 Aktie {i + 1}", expanded=(i < 3)):
            c_name, c_price, c_sector = st.columns([2, 1, 1])
            name = c_name.text_input(
                "Name / Ticker", value=f"Aktie {i+1}", key=f"a{i}_name",
            )
            price = c_price.number_input(
                "Aktueller Kurs (€)", value=50.0, min_value=0.01,
                format="%.2f", key=f"a{i}_price",
            )
            sector = c_sector.selectbox(
                "Sektor", SECTOR_LIST, key=f"a{i}_sector",
            )

            st.markdown("**Fair-Value-Verteilung**")
            dist_type = st.selectbox(
                "Verteilungstyp", DIST_OPTIONS, key=f"a{i}_dist",
            )

            params: dict = {}
            if dist_type == "Aus DCF-App (μ, σ, Schiefe)":
                st.caption(
                    "📊 Übertragen Sie die Werte aus dem Tab **Ergebnisse** der "
                    "SOTP-DCF-App (Abschnitt *Verteilungsparameter für Portfolio-App*)."
                )
                dc1, dc2, dc3 = st.columns(3)
                params["mean"] = dc1.number_input(
                    "μ (Mittelwert)", value=round(price * 1.2, 2),
                    min_value=0.01, format="%.4f", key=f"a{i}_dcf_mu",
                )
                params["std"] = dc2.number_input(
                    "σ (Std.-Abw.)", value=round(price * 0.2, 2),
                    min_value=0.01, format="%.4f", key=f"a{i}_dcf_sigma",
                )
                params["skew"] = dc3.number_input(
                    "Schiefe (Skew)", value=0.0,
                    min_value=-5.0, max_value=5.0,
                    format="%.4f", key=f"a{i}_dcf_skew",
                )
                if abs(params["skew"]) < 0.5:
                    st.success(
                        f"→ **Normalverteilung** wird verwendet "
                        f"(Schiefe {params['skew']:.3f} ≈ 0)"
                    )
                else:
                    st.info(
                        f"→ **Lognormalverteilung** wird verwendet "
                        f"(Schiefe {params['skew']:.3f} ≠ 0)"
                    )
            elif dist_type in ("Normal", "Lognormal"):
                dc1, dc2 = st.columns(2)
                params["mean"] = dc1.number_input(
                    "Ø Fair Value (€)", value=round(price * 1.2, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_mean",
                )
                params["std"] = dc2.number_input(
                    "Std.-Abw. σ (€)", value=round(price * 0.2, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_std",
                )
            elif dist_type in ("PERT", "Dreiecksverteilung"):
                dc1, dc2, dc3 = st.columns(3)
                params["low"] = dc1.number_input(
                    "Worst Case (€)", value=round(price * 0.7, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_lo",
                )
                params["mode"] = dc2.number_input(
                    "Base Case (€)", value=round(price * 1.2, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_mode",
                )
                params["high"] = dc3.number_input(
                    "Best Case (€)", value=round(price * 1.6, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_hi",
                )
            elif dist_type == "Gleichverteilung":
                dc1, dc2 = st.columns(2)
                params["low"] = dc1.number_input(
                    "Minimum (€)", value=round(price * 0.7, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_ulo",
                )
                params["high"] = dc2.number_input(
                    "Maximum (€)", value=round(price * 1.5, 2),
                    min_value=0.01, format="%.2f", key=f"a{i}_fv_uhi",
                )

            # Weight constraints
            with st.container():
                st.markdown("**Optionale Einschränkungen**")
                oc1, oc2 = st.columns(2)
                min_weight = oc1.number_input(
                    "Min. Gewicht (%)", value=0.0, min_value=0.0,
                    max_value=100.0, format="%.1f", key=f"a{i}_wmin",
                )
                max_weight = oc2.number_input(
                    "Max. Gewicht (%)", value=100.0, min_value=0.0,
                    max_value=100.0, format="%.1f", key=f"a{i}_wmax",
                )

            asset_configs.append({
                "name": name,
                "price": price,
                "sector": sector,
                "dist_type": dist_type,
                "params": params,
                "min_weight": min_weight / 100.0,
                "max_weight": max_weight / 100.0,
            })

    # ── Korrelationsmatrix ────────────────────────────────────────────
    st.divider()
    st.subheader("📐 Korrelationsmatrix")

    with st.expander("ℹ️ Warum ist die Korrelation wichtig?", expanded=False):
        st.markdown("""
Die **Korrelation** zwischen den Assets ist der Schlüssel zur Diversifikation:

| Korrelation ρ | Bedeutung | Portfolioeffekt |
|---|---|---|
| +1.0 | Perfekt gleichgerichtet | Keine Diversifikation |
| +0.5 bis +0.8 | Typisch gleiche Branche | Geringe Diversifikation |
| +0.2 bis +0.5 | Typisch verschiedene Branchen | Gute Diversifikation |
| 0.0 | Unkorreliert | Sehr gute Diversifikation |
| −0.5 bis 0.0 | Gegenläufig | Exzellente Diversifikation |

> **Hinweis:** In Krisen steigen Korrelationen oft stark an (ρ → 1).
> Für konservative Analysen: Korrelationen eher hoch ansetzen.

**Cluster-basiertes Modell (Standard):**
Das Tool verwendet ein verfeinertes Sektormodell, das Sektoren nach
wirtschaftlichen Clustern gruppiert (Growth, Cyclical, Defensive,
Financial, Energy). Gleicher Sektor = ρ ≈ 0.65, ähnliches Cluster ≈ 0.45–0.55,
verschiedene Cluster ≈ 0.15–0.40.
""")

    corr_method = st.radio(
        "Korrelationsquelle",
        ["Cluster-basiert (nach Sektor)", "Manuell eingeben", "Unkorreliert (ρ = 0)"],
        horizontal=True,
    )

    n_total = len(asset_configs)
    sectors = [ac["sector"] for ac in asset_configs]

    if corr_method == "Cluster-basiert (nach Sektor)":
        corr_matrix = PortfolioAnalyser.build_sector_correlation(sectors)
        st.info(
            "**Cluster-basierte Korrelationen**: Gleicher Sektor ≈ 0.65 · "
            "Gleiches Cluster ≈ 0.45–0.55 · Verschiedene Cluster ≈ 0.15–0.40"
        )
    elif corr_method == "Manuell eingeben":
        corr_matrix = np.eye(n_total)
        for i_c in range(n_total):
            for j_c in range(i_c + 1, n_total):
                val = st.number_input(
                    f"ρ({asset_configs[i_c]['name']}, {asset_configs[j_c]['name']})",
                    min_value=-1.0, max_value=1.0, value=0.3,
                    format="%.2f", key=f"corr_{i_c}_{j_c}",
                )
                corr_matrix[i_c, j_c] = val
                corr_matrix[j_c, i_c] = val
        # Enforce positive semi-definiteness
        corr_matrix = PortfolioAnalyser.ensure_psd(corr_matrix)
        st.caption("✅ Korrelationsmatrix wurde auf positive Semi-Definitheit geprüft (PSD-Projektion).")
    else:
        corr_matrix = np.eye(n_total)

    # Display correlation matrix
    if n_total >= 2:
        names_list = [a["name"] for a in asset_configs]
        corr_df = pd.DataFrame(corr_matrix, index=names_list, columns=names_list)
        st.dataframe(
            corr_df.style.format("{:.2f}").background_gradient(
                cmap="RdYlGn", vmin=-1, vmax=1,
            ),
            use_container_width=True,
        )

    # ── Save config as JSON ───────────────────────────────────────────
    st.divider()

    save_col, run_col = st.columns([1, 2])
    with save_col:
        config_json = json.dumps({
            "assets": asset_configs,
            "corr_method": corr_method,
            "risk_free_pct": risk_free_pct,
            "n_mc_sim": int(n_mc_sim),
            "seed": int(global_seed),
        }, indent=2, default=str)
        st.download_button(
            "💾 Konfiguration speichern",
            data=config_json,
            file_name="portfolio_config.json",
            mime="application/json",
        )

    # ── RUN ───────────────────────────────────────────────────────────
    with run_col:
        run_analysis = st.button(
            "🚀 Portfolio-Analyse starten",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        with st.spinner("Generiere Fair-Value-Verteilungen & optimiere Portfolio …"):
            # 1) Generate FV samples & build AssetInput objects
            asset_inputs: list[AssetInput] = []
            for idx, ac in enumerate(asset_configs):
                fv = generate_fv_samples(
                    ac["dist_type"], ac["params"],
                    n=int(n_mc_sim),
                    seed=int(global_seed) + idx,
                )
                asset_inputs.append(AssetInput(
                    name=ac["name"],
                    sector=ac["sector"],
                    current_price=ac["price"],
                    fv_samples=fv,
                    min_weight=ac["min_weight"],
                    max_weight=ac["max_weight"],
                ))

            # 2) Instantiate analyser
            analyser = PortfolioAnalyser(risk_free_rate=risk_free_pct / 100.0)

            # 3) Single-asset analysis
            asset_metrics = analyser.analyse_all(asset_inputs)

            # 4) Build matrices
            returns_matrix = analyser.build_returns_matrix(asset_inputs)
            mu_vec, std_vec, cov_matrix = analyser.build_covariance(
                returns_matrix, corr_matrix,
            )

            # 5) Run all 7 optimisations
            bounds = [(ai.min_weight, ai.max_weight) for ai in asset_inputs]
            opt_results = analyser.run_all_optimisations(
                asset_metrics, mu_vec, cov_matrix, std_vec, returns_matrix, bounds,
            )

            # 6) Efficient frontier
            if n_total >= 2:
                ef_vols, ef_rets = analyser.efficient_frontier(
                    mu_vec, cov_matrix, bounds,
                )
            else:
                ef_vols, ef_rets = np.array([]), np.array([])

            # Store everything in session state
            st.session_state.pf_results = {
                "asset_metrics": asset_metrics,
                "asset_inputs": asset_inputs,
                "corr_matrix": corr_matrix,
                "cov_matrix": cov_matrix,
                "mu_vec": mu_vec,
                "std_vec": std_vec,
                "returns_matrix": returns_matrix,
                "rf": risk_free_pct / 100.0,
                "opt_results": opt_results,
                "ef_vols": ef_vols,
                "ef_rets": ef_rets,
                "n_sim": int(n_mc_sim),
                "sectors": sectors,
            }

        st.success(
            f"✅ Analyse abgeschlossen – {n_total} Assets · "
            f"{int(n_mc_sim):,} MC-Simulationen · 7 Optimierungsmethoden"
        )
        st.balloons()


# ══════════════════════════════════════════════════════════════════════════
# Helper: collect active optimisation results in display order
# ══════════════════════════════════════════════════════════════════════════

def _active_results(pf: dict) -> OrderedDict[str, PortfolioResult]:
    """Return non-None results in the canonical display order."""
    out = OrderedDict()
    for key in METHOD_ORDER:
        res = pf["opt_results"].get(key)
        if res is not None:
            out[key] = res
    return out


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 – EINZELTITEL-ANALYSE
# ──────────────────────────────────────────────────────────────────────────

with tab_single:
    if st.session_state.pf_results is None:
        st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
    else:
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
                # Metrics
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

                # Charts
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

                # Interpretation
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


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 – PORTFOLIO-OPTIMIERUNG
# ──────────────────────────────────────────────────────────────────────────

with tab_portfolio:
    if st.session_state.pf_results is None:
        st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
    else:
        pf = st.session_state.pf_results
        st.header("📊 Portfolio-Optimierung")
        active = _active_results(pf)

        with st.expander("ℹ️ Erklärung der 7 Optimierungsmethoden", expanded=False):
            st.markdown(r"""
**1. Gleichgewicht (1/N)** – Naives Benchmark.  Überraschend robust,
da keine Schätzfehler einfließen.

**2. Max Sharpe Ratio** – $\max_w \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}}$
– höchste risikoadjustierte Rendite, aber sensitiv gegenüber Inputs.

**3. Min Volatilität** – $\min_w \sqrt{w^T\Sigma w}$ – minimales
Gesamtrisiko.  Stabiler als Max Sharpe.

**4. Risk Parity** – Jedes Asset trägt *gleich viel Risiko*.  Beliebt
bei institutionellen Investoren (z.B. Bridgewater All Weather).

**5. Min CVaR (Expected Shortfall)** – Minimiert den *erwarteten Verlust
im schlimmsten 5%-Tail* der Monte-Carlo-Verteilung.  Robuster als
Varianzminimierung bei Tail-Risiken. *(Neu)*

**6. Max Diversifikation** – Maximiert das Diversifikationsratio
$DR = \frac{\sum w_i \sigma_i}{\sigma_P}$.  Verteilt risiko-optimal
über möglichst unkorrelierte Assets. *(Neu)*

**7. Kelly (Multi-Asset)** – Maximiert
$w^T\mu - \frac{1}{2} w^T \Sigma w$ (erwartetes Log-Wachstum) mit
Half-Kelly-Skalierung.  Berücksichtigt Kovarianzen zwischen Assets
(Verbesserung: Multi-Asset statt Einzel-Kelly). *(Verbessert)*
""")

        names = [am.name for am in pf["asset_metrics"]]

        # ── Weights comparison table ──────────────────────────────────
        st.subheader("⚖️ Gewichtungsvergleich")

        weights_dict: dict[str, np.ndarray] = {}
        weights_table_data = {"Asset": names}
        for method_name, pr in active.items():
            weights_table_data[method_name] = [f"{w:.1%}" for w in pr.weights]
            weights_dict[method_name] = pr.weights

        st.dataframe(
            pd.DataFrame(weights_table_data),
            use_container_width=True, hide_index=True,
        )

        # ── Weights bar chart ─────────────────────────────────────────
        st.divider()
        st.plotly_chart(
            portfolio_weights_comparison(names, weights_dict),
            use_container_width=True,
        )

        # ── Portfolio metrics per method ──────────────────────────────
        st.divider()
        st.subheader("📊 Portfolio-Kennzahlen je Methode")

        metrics_rows = []
        for pr in active.values():
            metrics_rows.append({
                "Methode": pr.name,
                "E[Rendite]": f"{pr.expected_return:+.1%}",
                "Volatilität": f"{pr.volatility:.1%}",
                "Sharpe Ratio": f"{pr.sharpe_ratio:.2f}",
                "VaR (5%)": f"{pr.var_5:+.1%}",
                "CVaR (5%)": f"{pr.cvar_5:+.1%}",
                "P(Verlust)": f"{pr.prob_loss:.1%}",
                "Div.-Ratio": f"{pr.diversification_ratio:.2f}",
                "Eff. # Assets": f"{pr.effective_n_assets:.1f}",
            })

        st.dataframe(
            pd.DataFrame(metrics_rows),
            use_container_width=True, hide_index=True,
        )

        # ── Portfolio return distribution ─────────────────────────────
        st.divider()
        st.subheader("📈 Portfolio-Renditeverteilung")

        selected_method = st.selectbox(
            "Methode auswählen",
            list(active.keys()),
            key="pf_method_select",
        )

        sel_pr = active[selected_method]
        port_ret = pf["returns_matrix"] @ sel_pr.weights
        port_ret_pct = port_ret * 100

        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(
                histogram_kde(
                    port_ret_pct,
                    f"Renditeverteilung – {selected_method}",
                    "Portfolio-Rendite (%)",
                    color=COLORS["accent"],
                    vlines={"Breakeven": (0, COLORS["neutral"], "solid")},
                    show_percentile_lines=False,
                ),
                use_container_width=True,
            )
        with pc2:
            st.plotly_chart(
                cdf_with_reference(
                    port_ret_pct,
                    f"CDF – {selected_method}",
                    "Portfolio-Rendite (%)",
                    ref_value=0,
                    ref_label="Breakeven",
                ),
                use_container_width=True,
            )

        # ── Diversifikationsanalyse ───────────────────────────────────
        if len(names) >= 2:
            st.divider()
            st.subheader("🔀 Diversifikationsanalyse")

            with st.expander("ℹ️ Was bedeuten diese Kennzahlen?", expanded=False):
                st.markdown(r"""
**Diversifikationsratio** $DR = \frac{\sum_i w_i \sigma_i}{\sigma_P}$
- DR = 1.0: Keine Diversifikation · DR > 1.5: Gute Diversifikation

**Effektive Anzahl Assets** $N_{eff} = \frac{1}{\sum_i w_i^2}$
- $N_{eff}$ = N: Perfekt gleichgewichtet · $N_{eff}$ > 5: Gut diversifiziert
""")

            div_rows = []
            for pr in active.values():
                div_rows.append({
                    "Methode": pr.name,
                    "Diversifikationsratio": f"{pr.diversification_ratio:.2f}",
                    "Effektive # Assets": f"{pr.effective_n_assets:.1f}",
                })

            st.dataframe(
                pd.DataFrame(div_rows),
                use_container_width=True, hide_index=True,
            )


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 – EFFICIENT FRONTIER
# ──────────────────────────────────────────────────────────────────────────

with tab_frontier:
    if st.session_state.pf_results is None:
        st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
    elif len(st.session_state.pf_results["asset_metrics"]) < 2:
        st.info("Die Efficient Frontier benötigt mindestens 2 Assets.")
    else:
        pf = st.session_state.pf_results
        active = _active_results(pf)
        st.header("📈 Efficient Frontier")

        with st.expander("ℹ️ Was ist die Efficient Frontier?", expanded=False):
            st.markdown(r"""
Die Efficient Frontier zeigt alle Portfolios mit **maximaler Rendite
für ein gegebenes Risiko** (oder minimalem Risiko für gegebene Rendite).

**Capital Market Line (CML):**
$E[R_P] = r_f + \frac{E[R_M] - r_f}{\sigma_M} \cdot \sigma_P$

Portfolios *auf* der Frontier sind **effizient** – kein anderes Portfolio
bietet bei gleichem Risiko mehr Rendite.
""")

        names = [am.name for am in pf["asset_metrics"]]
        fig_ef = go.Figure()

        # Frontier curve
        if len(pf["ef_vols"]) > 0:
            fig_ef.add_trace(go.Scatter(
                x=pf["ef_vols"] * 100, y=pf["ef_rets"] * 100,
                mode="lines", name="Efficient Frontier",
                line=dict(color=COLORS["primary"], width=3),
            ))

        # Individual assets
        for idx, am in enumerate(pf["asset_metrics"]):
            fig_ef.add_trace(go.Scatter(
                x=[am.return_std * 100],
                y=[am.expected_return * 100],
                mode="markers+text",
                name=am.name,
                marker=dict(
                    size=12,
                    color=PALETTE_EXTENDED[idx % len(PALETTE_EXTENDED)],
                ),
                text=[am.name],
                textposition="top center",
            ))

        # Optimised portfolio points
        SYMBOLS = {
            "Gleichgewicht (1/N)": "circle",
            "Max Sharpe": "star",
            "Min Volatilität": "diamond",
            "Risk Parity": "square",
            "Min CVaR": "hexagon",
            "Max Diversifikation": "cross",
            "Kelly (Multi-Asset)": "pentagon",
        }
        POINT_COLORS = {
            "Gleichgewicht (1/N)": COLORS["neutral"],
            "Max Sharpe": COLORS["primary"],
            "Min Volatilität": "#17becf",
            "Risk Parity": COLORS["secondary"],
            "Min CVaR": COLORS["negative"],
            "Max Diversifikation": COLORS["positive"],
            "Kelly (Multi-Asset)": COLORS["accent"],
        }

        for method_name, pr in active.items():
            ret = pr.expected_return * 100
            vol = pr.volatility * 100
            fig_ef.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                name=method_name,
                marker=dict(
                    size=16,
                    color=POINT_COLORS.get(method_name, COLORS["accent"]),
                    symbol=SYMBOLS.get(method_name, "circle"),
                    line=dict(width=2, color="white"),
                ),
                text=[method_name],
                textposition="bottom center",
                textfont=dict(size=9),
            ))

        # Capital Market Line
        ms = active.get("Max Sharpe")
        if ms is not None and ms.volatility > 1e-12:
            slope = (ms.expected_return - pf["rf"]) / ms.volatility
            x_max = max(pf["std_vec"].max() * 1.2, ms.volatility * 1.5)
            x_cml = np.linspace(0, x_max, 50)
            y_cml = pf["rf"] + slope * x_cml
            fig_ef.add_trace(go.Scatter(
                x=x_cml * 100, y=y_cml * 100,
                mode="lines", name="Capital Market Line",
                line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
            ))

        fig_ef.update_layout(
            title="Efficient Frontier & Portfolio-Positionen",
            xaxis_title="Volatilität (%)",
            yaxis_title="Erwartete Rendite (%)",
            template=TEMPLATE, height=600,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # ── Correlation heatmap ───────────────────────────────────────
        st.divider()
        st.subheader("🔥 Korrelationsmatrix (Heatmap)")
        st.plotly_chart(
            correlation_heatmap(pf["corr_matrix"], names),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 – STRESS-TESTS
# ──────────────────────────────────────────────────────────────────────────

with tab_stress:
    if st.session_state.pf_results is None:
        st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
    else:
        pf = st.session_state.pf_results
        active = _active_results(pf)
        st.header("⚡ Stress-Tests & Szenario-Analyse")

        with st.expander("ℹ️ Warum Stress-Tests?", expanded=False):
            st.markdown("""
Stress-Tests prüfen die **Robustheit** Ihres Portfolios.

| Szenario | Beispiel |
|---|---|
| Marktcrash −30 % | COVID-19 (Feb–Mar 2020) |
| Sektorkrise −50 % | Bankenaktien 2008/09 |
| Korrelationsanstieg → 0.9 | Lehman-Krise 2008 |

> **Ziel:** Kein Portfolio sollte unter realistischen Stress-Szenarien
> zu einem untragbaren Verlust führen.
""")

        # ── Presets ───────────────────────────────────────────────────
        st.subheader("🎛️ Stress-Szenario konfigurieren")

        preset = st.selectbox(
            "Preset wählen",
            ["Benutzerdefiniert", "COVID-19 Crash", "GFC 2008", "Mild Correction"],
            key="stress_preset",
        )

        preset_defaults = {
            "COVID-19 Crash": (-35, 0.90, -20),
            "GFC 2008": (-50, 0.95, -40),
            "Mild Correction": (-15, 0.70, -10),
            "Benutzerdefiniert": (-30, 0.85, -20),
        }
        p_market, p_corr, p_sector = preset_defaults[preset]

        sc1, sc2 = st.columns(2)
        with sc1:
            market_shock = st.slider(
                "Marktschock (%)", min_value=-80, max_value=0,
                value=p_market,
            )
        with sc2:
            corr_stress = st.slider(
                "Korrelations-Stress (min ρ):",
                min_value=0.0, max_value=1.0, value=p_corr, step=0.05,
            )

        sectors_in_portfolio = list(set(
            am.sector for am in pf["asset_metrics"]
        ))
        shock_sector = st.selectbox(
            "Sektor-Schock (optional)",
            ["Keiner"] + sectors_in_portfolio,
        )
        sector_shock_pct = 0
        if shock_sector != "Keiner":
            sector_shock_pct = st.slider(
                f"Zusätzlicher Schock – {shock_sector} (%)",
                min_value=-80, max_value=0, value=p_sector,
            )

        st.divider()

        if st.button("⚡ Stress-Test durchführen", type="primary",
                      use_container_width=True):

            analyser = PortfolioAnalyser(risk_free_rate=pf["rf"])

            # Build weight dict for stress test
            portfolio_weights = {
                name: pr.weights for name, pr in active.items()
            }

            stress_results = analyser.stress_test(
                portfolios=portfolio_weights,
                returns_matrix=pf["returns_matrix"],
                asset_sectors=pf["sectors"],
                market_shock_pct=float(market_shock),
                corr_stress=float(corr_stress),
                sector_shock=(
                    shock_sector if shock_sector != "Keiner" else None
                ),
                sector_shock_pct=float(sector_shock_pct),
            )

            # ── Results table ─────────────────────────────────────────
            st.subheader("📊 Stress-Test-Ergebnisse")

            stress_rows = []
            for sr in stress_results:
                stress_rows.append({
                    "Methode": sr.method_name,
                    "E[R] Normal": f"{sr.return_normal:+.1%}",
                    "E[R] Stress": f"{sr.return_stressed:+.1%}",
                    "Δ Rendite": f"{sr.delta_return:+.1%}",
                    "Vol (Stress)": f"{sr.vol_stressed:.1%}",
                    "VaR 5%": f"{sr.var_5_stressed:+.1%}",
                    "CVaR 5%": f"{sr.cvar_5_stressed:+.1%}",
                    "P(Verlust)": f"{sr.prob_loss:.1%}",
                })

            st.dataframe(
                pd.DataFrame(stress_rows),
                use_container_width=True, hide_index=True,
            )

            # ── Stress chart ──────────────────────────────────────────
            st.divider()
            st.subheader("📈 Normal vs. Stress-Szenario")

            sel_stress_method = st.selectbox(
                "Methode für Detail-Ansicht",
                [sr.method_name for sr in stress_results],
                key="stress_method_sel",
            )

            sel_w = active[sel_stress_method].weights

            shocked_returns = pf["returns_matrix"] + (market_shock / 100.0)
            if shock_sector != "Keiner" and sector_shock_pct != 0:
                for idx, sec in enumerate(pf["sectors"]):
                    if sec == shock_sector:
                        shocked_returns[:, idx] += (sector_shock_pct / 100.0)

            normal_port = pf["returns_matrix"] @ sel_w
            stressed_port = shocked_returns @ sel_w

            st.plotly_chart(
                stress_comparison_chart(
                    normal_port, stressed_port, sel_stress_method,
                ),
                use_container_width=True,
            )

            # ── Interpretation ────────────────────────────────────────
            st.divider()
            st.subheader("💡 Stress-Test Interpretation")

            best = min(stress_results, key=lambda x: x.cvar_5_stressed)
            worst = max(stress_results, key=lambda x: x.prob_loss)

            st.markdown(f"""
**Szenario:** Marktschock **{market_shock}%** · Korrelation ≥ **{corr_stress:.2f}**
{f" · Sektorschock {shock_sector} **{sector_shock_pct}%**" if shock_sector != "Keiner" else ""}

- **Robustestes Portfolio** (niedrigstes CVaR): **{best.method_name}**
  – CVaR (5%) = {best.cvar_5_stressed:+.1%}
- **Höchste Verlustwahrscheinlichkeit**: **{worst.method_name}**
  – P(Verlust) = {worst.prob_loss:.1%}

> **Faustregel:** CVaR (5%) unter Stress > −40% → Konzentration
> reduzieren oder defensivere Assets beimischen.
""")
