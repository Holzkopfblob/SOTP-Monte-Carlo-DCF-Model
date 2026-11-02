"""
Reusable Streamlit UI Helper Functions.

Contains the distribution-input renderer and informational
explanation boxes for financial / statistical concepts.
"""
from __future__ import annotations

import streamlit as st

from domain.models import DistributionConfig, DistributionType
from infrastructure.config_io import dist_config_from_state
from presentation.explanations import (
    text_corporate_bridge,
    text_distributions,
    text_fade_model,
    text_fcff,
    text_interpretation,
    text_monte_carlo,
    text_sotp,
    text_terminal_value,
    text_wacc,
)


# ═══════════════════════════════════════════════════════════════════════════
# Distribution type option labels (ordered)
# ═══════════════════════════════════════════════════════════════════════════

DIST_OPTIONS = [dt.value for dt in DistributionType]


def render_source(key: str) -> None:
    """Show the research justification for *key* from the loaded ``quellen`` block."""
    quellen = (st.session_state.get("cfg_passthrough") or {}).get("quellen") or {}
    if quellen.get(key):
        st.caption(f"📎 {quellen[key]}")


# ═══════════════════════════════════════════════════════════════════════════
# Core input renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_distribution_input(
    label: str,
    key_prefix: str,
    default_value: float = 0.0,
    is_percentage: bool = False,
    help_text: str = "",
) -> DistributionConfig:
    """
    Render Streamlit input widgets for one stochastic parameter.

    Parameters
    ----------
    label : str
        Human-readable parameter name.
    key_prefix : str
        Globally unique prefix for widget keys.
    default_value : float
        Default *display* value.  If ``is_percentage`` is True this is in
        % (e.g. 5 for 5 %).  Internally we store decimal fractions.
    is_percentage : bool
        If True, all user-facing values are interpreted as % and divided
        by 100 before storage.
    help_text : str
        Optional short explanation shown as caption.

    Returns
    -------
    DistributionConfig
        Fully populated config with decimal values.
    """
    pct_label = " (%)" if is_percentage else ""
    d = default_value  # shorthand for display default

    # ── Header row ────────────────────────────────────────────────────
    col_lbl, col_dist = st.columns([2, 3])
    with col_lbl:
        st.markdown(f"**{label}{pct_label}**")
        if help_text:
            st.caption(help_text)
        render_source(key_prefix)
    with col_dist:
        dist_type_str = st.selectbox(
            "Verteilung",
            options=DIST_OPTIONS,
            key=f"{key_prefix}_dtype",
            label_visibility="collapsed",
        )

    dist_type = DistributionType(dist_type_str)

    # ── Parameter inputs (conditional) ────────────────────────────────
    # Widgets only; their values land in st.session_state under the keys
    # below and are converted by config_io (shared with headless runs).
    if dist_type == DistributionType.FIXED:
        st.number_input(
            f"Wert{pct_label}",
            value=d,
            key=f"{key_prefix}_fixed",
            format="%.4f",
        )

    elif dist_type == DistributionType.NORMAL:
        c1, c2 = st.columns(2)
        c1.number_input(
            "μ (Mittelwert)", value=d,
            key=f"{key_prefix}_n_mu", format="%.4f",
        )
        sigma = c2.number_input(
            "σ (Std.-Abw.)", value=max(abs(d) * 0.2, 0.01),
            key=f"{key_prefix}_n_sig", format="%.4f",
        )
        if sigma <= 0:
            st.warning("⚠️ Standardabweichung σ muss > 0 sein. Wert wird auf Minimum gesetzt.")

    elif dist_type == DistributionType.LOGNORMAL:
        st.caption(
            "Geben Sie den **gewünschten** Mittelwert und die Std.-Abw. "
            "der Lognormalverteilung ein (nicht die der zugrunde liegenden "
            "Normalverteilung). Die Konvertierung erfolgt automatisch."
        )
        c1, c2 = st.columns(2)
        mu_disp = c1.number_input(
            "Gewünschter Mittelwert", value=max(d, 0.01),
            key=f"{key_prefix}_ln_mu", format="%.4f",
        )
        sig_disp = c2.number_input(
            "Gewünschte Std.-Abw.", value=max(abs(d) * 0.2, 0.01),
            key=f"{key_prefix}_ln_sig", format="%.4f",
        )
        if mu_disp <= 0:
            st.warning("⚠️ Lognormalverteilung erfordert einen positiven Mittelwert.")
        if sig_disp <= 0:
            st.warning("⚠️ Standardabweichung muss > 0 sein.")

    elif dist_type == DistributionType.TRIANGULAR:
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_tri_lo", format="%.4f")
        mo = c2.number_input("Mode", value=d, key=f"{key_prefix}_tri_mo", format="%.4f")
        hi = c3.number_input("Max", value=d * 1.2, key=f"{key_prefix}_tri_hi", format="%.4f")
        if lo >= hi:
            st.warning("⚠️ Min muss kleiner als Max sein. Werte werden automatisch korrigiert.")
        if mo < lo or mo > hi:
            st.warning("⚠️ Mode sollte zwischen Min und Max liegen. Wird auf den gültigen Bereich geklammert.")

    elif dist_type == DistributionType.UNIFORM:
        c1, c2 = st.columns(2)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_uni_lo", format="%.4f")
        hi = c2.number_input("Max", value=d * 1.2, key=f"{key_prefix}_uni_hi", format="%.4f")
        if lo >= hi:
            st.warning("⚠️ Min muss kleiner als Max sein. Werte werden automatisch korrigiert.")

    elif dist_type == DistributionType.PERT:
        st.caption(
            "PERT-Verteilung: Mehr Gewicht auf dem wahrscheinlichsten Wert "
            "als die Dreiecksverteilung – ideal für Expertenschätzungen."
        )
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Min", value=d * 0.7, key=f"{key_prefix}_pert_lo", format="%.4f")
        mo = c2.number_input("Mode", value=d, key=f"{key_prefix}_pert_mo", format="%.4f")
        hi = c3.number_input("Max", value=d * 1.3, key=f"{key_prefix}_pert_hi", format="%.4f")
        if lo >= hi:
            st.warning("⚠️ Min muss kleiner als Max sein. Werte werden automatisch korrigiert.")
        if mo < lo or mo > hi:
            st.warning("⚠️ Mode sollte zwischen Min und Max liegen. Wird auf den gültigen Bereich geklammert.")

    return dist_config_from_state(st.session_state, key_prefix, d, is_percentage)


# ═══════════════════════════════════════════════════════════════════════════
# Concept explanation boxes – generic renderer + convenience aliases
# ═══════════════════════════════════════════════════════════════════════════

# Registry mapping short keys to (expander title, text factory function).
# Adding a new info box only requires a new entry here.
_INFO_REGISTRY: dict[str, tuple[str, callable]] = {
    "fcff":             ("ℹ️ Was ist FCFF (Free Cash Flow to Firm)?", text_fcff),
    "wacc":             ("ℹ️ Was ist WACC? (ausführlich)",            text_wacc),
    "distributions":    ("ℹ️ Wahrscheinlichkeitsverteilungen – Wann welche nutzen?", text_distributions),
    "terminal_value":   ("ℹ️ Terminal Value – Methoden (ausführlich)", text_terminal_value),
    "monte_carlo":      ("ℹ️ Was ist eine Monte-Carlo-Simulation?",   text_monte_carlo),
    "corporate_bridge": ("ℹ️ Unternehmensbrücke – vom EV zum Equity Value", text_corporate_bridge),
    "interpretation":   ("ℹ️ Wie lese ich die Ergebnisse richtig?",   text_interpretation),
    "sotp":             ("ℹ️ Was ist Sum-of-the-Parts (SOTP)?",       text_sotp),
    "fade_model":       ("ℹ️ Revenue-Fade-Modell – Konvergierendes Wachstum", text_fade_model),
}


def render_info_box(key: str) -> None:
    """Render an expandable info box by registry key.

    >>> render_info_box("fcff")
    """
    title, text_fn = _INFO_REGISTRY[key]
    with st.expander(title):
        st.markdown(text_fn())


# Backward-compatible short-hand aliases (one-liners that delegate)
def render_info_fcff()             -> None: render_info_box("fcff")
def render_info_wacc()             -> None: render_info_box("wacc")
def render_info_distributions()    -> None: render_info_box("distributions")
def render_info_terminal_value()   -> None: render_info_box("terminal_value")
def render_info_monte_carlo()      -> None: render_info_box("monte_carlo")
def render_info_corporate_bridge() -> None: render_info_box("corporate_bridge")
def render_info_interpretation()   -> None: render_info_box("interpretation")
def render_info_sotp()             -> None: render_info_box("sotp")
def render_info_fade_model()       -> None: render_info_box("fade_model")
