"""Correlation matrix input section for portfolio setup."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from application.portfolio_service import PortfolioAnalyser


def render_input_correlation_section(asset_configs: list[dict]) -> tuple[np.ndarray, list[str], str]:
    """Render correlation selection and return matrix, sectors and method label."""
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

    return corr_matrix, sectors, corr_method
