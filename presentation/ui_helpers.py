"""
Reusable Streamlit UI Helper Functions.

Contains the distribution-input renderer and informational
explanation boxes for financial / statistical concepts.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from domain.models import DistributionConfig, DistributionType
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
    divisor = 100.0 if is_percentage else 1.0
    d = default_value  # shorthand for display default

    # ── Header row ────────────────────────────────────────────────────
    col_lbl, col_dist = st.columns([2, 3])
    with col_lbl:
        st.markdown(f"**{label}{pct_label}**")
        if help_text:
            st.caption(help_text)
    with col_dist:
        dist_type_str = st.selectbox(
            "Verteilung",
            options=DIST_OPTIONS,
            key=f"{key_prefix}_dtype",
            label_visibility="collapsed",
        )

    dist_type = DistributionType(dist_type_str)
    config = DistributionConfig(dist_type=dist_type)

    # ── Parameter inputs (conditional) ────────────────────────────────
    if dist_type == DistributionType.FIXED:
        val = st.number_input(
            f"Wert{pct_label}",
            value=d,
            key=f"{key_prefix}_fixed",
            format="%.4f",
        )
        config.fixed_value = val / divisor

    elif dist_type == DistributionType.NORMAL:
        c1, c2 = st.columns(2)
        mu = c1.number_input(
            "μ (Mittelwert)", value=d,
            key=f"{key_prefix}_n_mu", format="%.4f",
        )
        sigma = c2.number_input(
            "σ (Std.-Abw.)", value=max(abs(d) * 0.2, 0.01),
            key=f"{key_prefix}_n_sig", format="%.4f",
        )
        if sigma <= 0:
            st.warning("⚠️ Standardabweichung σ muss > 0 sein. Wert wird auf Minimum gesetzt.")
        config.mean = mu / divisor
        config.std = sigma / divisor

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
        desired_mean = mu_disp / divisor
        desired_std = sig_disp / divisor
        # Convert displayed mean/std → underlying normal μ, σ
        if desired_mean > 1e-12 and desired_std > 1e-12:
            sigma_sq = float(np.log(1.0 + (desired_std / desired_mean) ** 2))
            config.ln_sigma = float(np.sqrt(sigma_sq))
            config.ln_mu = float(np.log(desired_mean) - sigma_sq / 2.0)
        else:
            config.ln_mu = float(np.log(max(desired_mean, 1e-12)))
            config.ln_sigma = 0.1

    elif dist_type == DistributionType.TRIANGULAR:
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_tri_lo", format="%.4f")
        mo = c2.number_input("Mode", value=d, key=f"{key_prefix}_tri_mo", format="%.4f")
        hi = c3.number_input("Max", value=d * 1.2, key=f"{key_prefix}_tri_hi", format="%.4f")
        if lo >= hi:
            st.warning("⚠️ Min muss kleiner als Max sein. Werte werden automatisch korrigiert.")
        if mo < lo or mo > hi:
            st.warning("⚠️ Mode sollte zwischen Min und Max liegen. Wird auf den gültigen Bereich geklammert.")
        config.low = lo / divisor
        config.mode = mo / divisor
        config.high = hi / divisor

    elif dist_type == DistributionType.UNIFORM:
        c1, c2 = st.columns(2)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_uni_lo", format="%.4f")
        hi = c2.number_input("Max", value=d * 1.2, key=f"{key_prefix}_uni_hi", format="%.4f")
        if lo >= hi:
            st.warning("⚠️ Min muss kleiner als Max sein. Werte werden automatisch korrigiert.")
        config.low = lo / divisor
        config.high = hi / divisor

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
        config.low = lo / divisor
        config.mode = mo / divisor
        config.high = hi / divisor

    return config


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
