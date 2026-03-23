"""
Portfolio Optimisation Tab – weights comparison & metrics
==========================================================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from presentation.layout.insights import render_summary_cards
from presentation.layout.states import render_warning_state
from presentation.charts import (
    COLORS,
    cdf_with_reference,
    histogram_kde,
    portfolio_robustness_panel,
    portfolio_weights_comparison,
)
from presentation.pages.pf_common import active_results
from presentation.pages.pf_sections.common_metrics import (
    build_portfolio_metrics_rows,
    build_portfolio_weights_table,
)


def render_portfolio(tab) -> None:
    """Render Tab 3 (Portfolio-Optimierung)."""
    with tab:
        if st.session_state.pf_results is None:
            render_warning_state("⚠️ Bitte zuerst Bewertungen eingeben und Analyse starten.")
            return

        pf = st.session_state.pf_results
        st.header("📊 Portfolio-Optimierung")
        active = active_results(pf)

        best_sharpe = max((pr.sharpe_ratio for pr in active.values()), default=float("nan"))
        best_prob_loss = min((pr.prob_loss for pr in active.values()), default=float("nan"))
        render_summary_cards([
            ("Aktive Methoden", f"{len(active)}", None),
            ("Assets", f"{len(pf['asset_metrics'])}", None),
            ("Beste Sharpe", f"{best_sharpe:.2f}", None),
            ("Niedrigste P(Verlust)", f"{best_prob_loss:.1%}", None),
        ])

        with st.expander("ℹ️ Erklärung der 9 Optimierungsmethoden", expanded=False):
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
Varianzminimierung bei Tail-Risiken.

**6. Max Diversifikation** – Maximiert das Diversifikationsratio
$DR = \frac{\sum w_i \sigma_i}{\sigma_P}$.  Verteilt risiko-optimal
über möglichst unkorrelierte Assets.

**7. Kelly (Multi-Asset)** – Maximiert
$w^T\mu - \frac{1}{2} w^T \Sigma w$ (erwartetes Log-Wachstum) mit
Half-Kelly-Skalierung.

**8. HRP (Hierarchical Risk Parity)** – Hierarchisches Clustering
auf der Korrelationsmatrix + rekursive Bisektionsallokation.
Benötigt **keine Matrixinversion** und ist robuster gegenüber
Schätzfehlern als Markowitz. *(Neu)*

**9. Black-Litterman** – Kombiniert Marktgleichgewichtsrenditen
mit subjektiven Analysteneinschätzungen:
$\mu_{BL} = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}q]$.
Nur aktiv, wenn Views definiert sind. *(Neu)*
""")

        names = [am.name for am in pf["asset_metrics"]]

        # ── Weights comparison table ──────────────────────────────────
        st.subheader("⚖️ Gewichtungsvergleich")

        weights_df, weights_dict = build_portfolio_weights_table(names, active)

        st.dataframe(
            weights_df,
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

        metrics_rows = build_portfolio_metrics_rows(active)

        st.dataframe(
            pd.DataFrame(metrics_rows),
            use_container_width=True, hide_index=True,
        )

        robustness_payload = [
            {"name": pr.name, "sharpe": pr.sharpe_ratio, "cvar": pr.cvar_5}
            for pr in active.values()
        ]
        if robustness_payload:
            st.plotly_chart(
                portfolio_robustness_panel(robustness_payload),
                use_container_width=True,
            )

        # ── Radar chart ───────────────────────────────────────────
        st.divider()
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
