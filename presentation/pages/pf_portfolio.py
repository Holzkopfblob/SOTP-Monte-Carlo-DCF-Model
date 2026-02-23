"""
Portfolio Optimisation Tab – weights comparison & metrics
==========================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from presentation.charts import (
    COLORS,
    cdf_with_reference,
    histogram_kde,
    portfolio_weights_comparison,
)
from presentation.pages.pf_common import active_results


def render_portfolio(tab) -> None:
    """Render Tab 3 (Portfolio-Optimierung)."""
    with tab:
        if st.session_state.pf_results is None:
            st.warning("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
            return

        pf = st.session_state.pf_results
        st.header("📊 Portfolio-Optimierung")
        active = active_results(pf)

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
        weights_table_data: dict[str, list] = {"Asset": names}
        for method_name, pr in active.items():
            weights_table_data[method_name] = [f"{w:.1%}" for w in pr.weights]
            weights_dict[method_name] = pr.weights

        st.dataframe(
            pd.DataFrame(weights_table_data),
            use_container_width=True, hide_index=True,
        )

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
