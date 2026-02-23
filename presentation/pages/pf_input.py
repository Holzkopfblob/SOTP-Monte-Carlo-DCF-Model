"""
Portfolio Input Tab – Asset configuration & analysis launch
=============================================================
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from application.portfolio_service import (
    AssetInput,
    PortfolioAnalyser,
    generate_fv_samples,
)
from presentation.pages.pf_common import DIST_OPTIONS, SECTOR_LIST


def render_input(tab, *, n_mc_sim: int, global_seed: int, risk_free_pct: float,
                 uploaded) -> None:
    """Render Tab 1 (Bewertungen eingeben)."""
    with tab:
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

        # ── Korrelationsmatrix ────────────────────────────────────────
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
            corr_matrix = PortfolioAnalyser.ensure_psd(corr_matrix)
            st.caption("✅ Korrelationsmatrix wurde auf positive Semi-Definitheit geprüft (PSD-Projektion).")
        else:
            corr_matrix = np.eye(n_total)

        if n_total >= 2:
            names_list = [a["name"] for a in asset_configs]
            corr_df = pd.DataFrame(corr_matrix, index=names_list, columns=names_list)
            st.dataframe(
                corr_df.style.format("{:.2f}").background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1,
                ),
                use_container_width=True,
            )

        # ── Save config as JSON ───────────────────────────────────────
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

        # ── RUN ───────────────────────────────────────────────────────
        with run_col:
            run_analysis = st.button(
                "🚀 Portfolio-Analyse starten",
                type="primary",
                use_container_width=True,
            )

        if run_analysis:
            with st.spinner("Generiere Fair-Value-Verteilungen & optimiere Portfolio …"):
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

                analyser = PortfolioAnalyser(risk_free_rate=risk_free_pct / 100.0)
                asset_metrics = analyser.analyse_all(asset_inputs)
                returns_matrix = analyser.build_returns_matrix(asset_inputs)
                mu_vec, std_vec, cov_matrix = analyser.build_covariance(
                    returns_matrix, corr_matrix,
                )
                bounds = [(ai.min_weight, ai.max_weight) for ai in asset_inputs]
                opt_results = analyser.run_all_optimisations(
                    asset_metrics, mu_vec, cov_matrix, std_vec, returns_matrix, bounds,
                )

                if n_total >= 2:
                    ef_vols, ef_rets = analyser.efficient_frontier(
                        mu_vec, cov_matrix, bounds,
                    )
                else:
                    ef_vols, ef_rets = np.array([]), np.array([])

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
