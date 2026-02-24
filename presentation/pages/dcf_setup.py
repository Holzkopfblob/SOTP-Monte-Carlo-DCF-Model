"""
DCF Setup Tab – Global simulation parameters & corporate bridge
================================================================
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from domain.models import DistributionConfig, SamplingMethod
from presentation.ui_helpers import (
    render_distribution_input,
    render_info_corporate_bridge,
    render_info_monte_carlo,
    render_info_sotp,
)


def render_setup(tab) -> dict:
    """Render Tab 1 (Setup) and return all configuration values as a dict.

    The returned dict contains every scalar/distribution needed by downstream
    tabs (segments, simulation).
    """
    with tab:
        st.header("Modell-Konfiguration")
        render_info_sotp()
        render_info_monte_carlo()

        col_a, col_b = st.columns(2)
        with col_a:
            n_simulations = st.number_input(
                "Anzahl Monte-Carlo-Iterationen",
                min_value=1_000, max_value=500_000, value=10_000, step=1_000,
                help="Mehr Iterationen → genauere Ergebnisse, längere Laufzeit.",
                key="setup_n_sim",
            )
            random_seed = st.number_input(
                "Random Seed (Reproduzierbarkeit)",
                value=42, min_value=0, key="setup_seed",
            )
        with col_b:
            n_segments = st.number_input(
                "Anzahl Geschäftssegmente",
                min_value=1, max_value=20, value=2,
                help="Für jedes Segment wird ein separater DCF berechnet.",
                key="setup_n_seg",
            )

        col_c, col_d = st.columns(2)
        with col_c:
            sampling_label = st.selectbox(
                "Sampling-Methode (Varianzreduktion)",
                [s.value for s in SamplingMethod],
                index=0,
                key="setup_sampling",
                help="Antithetic Variates halbiert die Varianz bei gleicher "
                     "Iterationsanzahl. Sobol (Quasi-MC) erzeugt gleichmäßigere "
                     "Abtastung des Parameterraums.",
            )
            sampling_method = SamplingMethod(sampling_label)

        st.markdown("")
        mid_year_conv = st.checkbox(
            "⏱️ Mid-Year Discounting Convention",
            value=True, key="setup_mid_year",
            help="Diskontiert FCFFs zur Jahresmitte (t−0,5) statt zum Jahresende. "
                 "Standardpraxis bei DCF-Bewertungen, da Cashflows unterjährig anfallen.",
        )

        st.divider()

        # ── Corporate bridge ──────────────────────────────────────────
        st.subheader("🏛️ Unternehmensbrücke (Corporate Bridge)")
        render_info_corporate_bridge()

        st.caption(
            "Jeder Bridge-Posten kann als **fester Wert** (deterministisch) "
            "oder als **Wahrscheinlichkeitsverteilung** (stochastisch) "
            "eingegeben werden. Wählen Sie einfach den Verteilungstyp – "
            "»Fest« entspricht einem einzelnen Punktschätzer."
        )

        st.markdown("##### Basis-Bridge")

        with st.expander("📐 Jährl. Holdingkosten (Mio.)", expanded=True):
            bridge_corp_costs = render_distribution_input(
                "Holdingkosten (Mio. p.a.)", "bridge_cc",
                default_value=50.0, is_percentage=False,
                help_text="Laufende Kosten der Holding-Gesellschaft p.a.",
            )

        with st.expander("📐 Diskontierung Holdingkosten (%)", expanded=True):
            bridge_corp_discount = render_distribution_input(
                "Diskontierungssatz (%)", "bridge_cd",
                default_value=9.0, is_percentage=True,
                help_text="Diskontierungssatz für die Perpetuity der Holdingkosten.",
            )

        with st.expander("📐 Nettoverschuldung (Mio.)", expanded=True):
            bridge_net_debt = render_distribution_input(
                "Nettoverschuldung (Mio.)", "bridge_nd",
                default_value=500.0, is_percentage=False,
                help_text="Finanzschulden − Cash & Äquivalente.",
            )

        with st.expander("📐 Aktien ausstehend (Mio.)", expanded=True):
            bridge_shares = render_distribution_input(
                "Aktien ausstehend (Mio.)", "bridge_sh",
                default_value=100.0, is_percentage=False,
                help_text="Voll verwässerte Aktienanzahl.",
            )

        # ── Extended Equity Bridge ────────────────────────────────────
        st.markdown("")
        st.markdown("##### Erweiterte Bridge")
        enable_ext_bridge = st.checkbox(
            "🏢 Erweiterte Equity Bridge aktivieren", value=False,
            key="setup_ext_bridge",
            help="Fügt zusätzliche Bridge-Posten hinzu: Minderheitsanteile, "
                 "Pensionsrückstellungen, nicht-operative Assets, Beteiligungen.",
        )

        bridge_minority: DistributionConfig | None = None
        bridge_pension: DistributionConfig | None = None
        bridge_non_op: DistributionConfig | None = None
        bridge_associates: DistributionConfig | None = None

        if enable_ext_bridge:
            st.caption(
                "Erweiterte Bridge-Posten für eine präzisere Equity-Value-Berechnung. "
                "Positive Werte bei Assets/Beteiligungen erhöhen, bei Verbindlichkeiten "
                "verringern sie den Equity Value."
            )
            with st.expander("📜 Minderheitsanteile (Mio.)", expanded=False):
                bridge_minority = render_distribution_input(
                    "Minderheitsanteile (Mio.)", "bridge_mi",
                    default_value=0.0, is_percentage=False,
                    help_text="Anteile Dritter an Tochtergesellschaften (wird abgezogen).",
                )
            with st.expander("📜 Pensionsrückstellungen (Mio.)", expanded=False):
                bridge_pension = render_distribution_input(
                    "Pensionsrückstellungen (Mio.)", "bridge_pn",
                    default_value=0.0, is_percentage=False,
                    help_text="Unterdeckung bei Pensionsverpflichtungen (wird abgezogen).",
                )
            with st.expander("📜 Nicht-operative Assets (Mio.)", expanded=False):
                bridge_non_op = render_distribution_input(
                    "Nicht-operative Assets (Mio.)", "bridge_no",
                    default_value=0.0, is_percentage=False,
                    help_text="Überschüssiges Cash, Immobilien, sonstige Investments (wird addiert).",
                )
            with st.expander("📜 Beteiligungen (Mio.)", expanded=False):
                bridge_associates = render_distribution_input(
                    "Beteiligungen (Mio.)", "bridge_as",
                    default_value=0.0, is_percentage=False,
                    help_text="Equity-Method Beteiligungen an assoziierten Unternehmen (wird addiert).",
                )

        # ── Cross-segment correlation (Phase 3) ──────────────────────
        st.divider()
        st.subheader("🔗 Segment-Korrelation (Cross-Segment)")

        segment_correlation: list[list[float]] | None = None

        enable_corr = st.checkbox(
            "Segment-Korrelation aktivieren (Gauss-Copula)",
            value=False, key="setup_corr_enable",
            help="Fügt stochastische Abhängigkeit zwischen den Segmenten "
                 "hinzu.  Ein hoher Korrelationswert bedeutet, dass gute / "
                 "schlechte Ergebnisse in verschiedenen Segmenten gemeinsam "
                 "auftreten.  Verwendet eine Gauss-Copula.",
        )
        if enable_corr and int(n_segments) >= 2:
            n_seg_int = int(n_segments)
            st.caption(
                f"Korrelationsmatrix ({n_seg_int}×{n_seg_int}) – Diagonal "
                "ist immer 1.  Geben Sie die paarweisen Korrelationen "
                "zwischen den Segmenten ein (−1 bis 1)."
            )
            corr_values: list[list[float]] = [
                [1.0] * n_seg_int for _ in range(n_seg_int)
            ]
            for row in range(n_seg_int):
                cols = st.columns(n_seg_int)
                for col_idx in range(n_seg_int):
                    if col_idx == row:
                        cols[col_idx].number_input(
                            f"ρ({row+1},{col_idx+1})",
                            value=1.0, disabled=True,
                            key=f"corr_{row}_{col_idx}",
                        )
                    elif col_idx > row:
                        val = cols[col_idx].number_input(
                            f"ρ({row+1},{col_idx+1})",
                            value=0.3, min_value=-1.0, max_value=1.0,
                            step=0.05, format="%.2f",
                            key=f"corr_{row}_{col_idx}",
                        )
                        corr_values[row][col_idx] = float(val)
                        corr_values[col_idx][row] = float(val)
                    else:
                        cols[col_idx].number_input(
                            f"ρ({row+1},{col_idx+1})",
                            value=float(corr_values[row][col_idx]),
                            disabled=True,
                            key=f"corr_{row}_{col_idx}",
                        )

            # Validate positive semi-definiteness
            corr_arr = np.array(corr_values)
            eigvals = np.linalg.eigvalsh(corr_arr)
            if np.any(eigvals < -1e-8):
                st.warning(
                    "⚠️ Die Matrix ist nicht positiv semi-definit.  "
                    "Bitte passen Sie die Korrelationswerte an."
                )
            else:
                segment_correlation = corr_values
        elif enable_corr and int(n_segments) < 2:
            st.info("Korrelation erfordert mindestens 2 Segmente.")

    return {
        "n_simulations": int(n_simulations),
        "random_seed": int(random_seed),
        "n_segments": int(n_segments),
        "mid_year_conv": bool(mid_year_conv),
        "sampling_method": sampling_method,
        "bridge_corp_costs": bridge_corp_costs,
        "bridge_corp_discount": bridge_corp_discount,
        "bridge_net_debt": bridge_net_debt,
        "bridge_shares": bridge_shares,
        "bridge_minority": bridge_minority,
        "bridge_pension": bridge_pension,
        "bridge_non_op": bridge_non_op,
        "bridge_associates": bridge_associates,
        "segment_correlation": segment_correlation,
    }
