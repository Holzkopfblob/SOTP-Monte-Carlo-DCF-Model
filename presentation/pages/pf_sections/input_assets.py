"""Asset and fair-value input section for portfolio setup."""
from __future__ import annotations

import streamlit as st

from presentation.pages.pf_common import PORTFOLIO_DIST_OPTIONS, SECTOR_LIST


def render_input_assets_section(n_assets: int) -> list[dict]:
    """Render asset-level inputs and return normalized asset configs."""
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
                "Verteilungstyp", PORTFOLIO_DIST_OPTIONS, key=f"a{i}_dist",
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

    return asset_configs
