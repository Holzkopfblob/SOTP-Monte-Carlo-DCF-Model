"""
Portfolio Input Tab – Asset configuration & analysis launch
=============================================================
"""
from __future__ import annotations

import json

import streamlit as st

from presentation.pages.pf_sections import (
    render_input_assets_section,
    render_input_correlation_section,
    render_input_covariance_section,
    render_input_run_section,
    render_input_views_section,
)


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

        asset_configs = render_input_assets_section(int(n_assets))
        corr_matrix, sectors, corr_method = render_input_correlation_section(asset_configs)
        _, cov_method = render_input_covariance_section()
        enable_bl, bl_views = render_input_views_section(asset_configs)
        render_input_run_section(
            asset_configs=asset_configs,
            corr_matrix=corr_matrix,
            corr_method=corr_method,
            sectors=sectors,
            cov_method=cov_method,
            n_mc_sim=int(n_mc_sim),
            global_seed=int(global_seed),
            risk_free_pct=risk_free_pct,
            enable_bl=enable_bl,
            bl_views=bl_views,
        )
