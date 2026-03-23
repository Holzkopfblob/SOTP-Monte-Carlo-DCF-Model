"""Corporate bridge section for DCF setup page."""

from __future__ import annotations

import streamlit as st

from domain.models import DistributionConfig
from presentation.ui_helpers import render_distribution_input


def render_setup_bridge_section() -> dict:
    """Render corporate bridge controls and return bridge distributions."""
    st.subheader("🏛️ Unternehmensbrücke (Corporate Bridge)")
    st.caption(
        "Jeder Bridge-Posten kann als **fester Wert** (deterministisch) "
        "oder als **Wahrscheinlichkeitsverteilung** (stochastisch) "
        "eingegeben werden. Wählen Sie einfach den Verteilungstyp - "
        "'Fest' entspricht einem einzelnen Punktschätzer."
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
            help_text="Finanzschulden - Cash & Äquivalente.",
        )

    with st.expander("📐 Aktien ausstehend (Mio.)", expanded=True):
        bridge_shares = render_distribution_input(
            "Aktien ausstehend (Mio.)", "bridge_sh",
            default_value=100.0, is_percentage=False,
            help_text="Voll verwässerte Aktienanzahl.",
        )

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

    st.divider()
    return {
        "bridge_corp_costs": bridge_corp_costs,
        "bridge_corp_discount": bridge_corp_discount,
        "bridge_net_debt": bridge_net_debt,
        "bridge_shares": bridge_shares,
        "bridge_minority": bridge_minority,
        "bridge_pension": bridge_pension,
        "bridge_non_op": bridge_non_op,
        "bridge_associates": bridge_associates,
    }
