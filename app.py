"""
SOTP Monte-Carlo DCF Simulation – Streamlit Application
========================================================

Main entry point.  Run with::

    streamlit run app.py

The application uses a four-step wizard, each step implemented in a
dedicated module under ``presentation/pages/``:

    1. Setup          – dcf_setup.py
    2. Segmente       – dcf_segments.py
    3. Simulation     – dcf_simulation.py
    4. Ergebnisse     – dcf_results.py
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import streamlit as st

from infrastructure.config_io import collect_config, apply_config
from presentation.layout.base import (
    configure_page,
    inject_global_styles,
    render_app_header,
    render_sidebar_footer,
)
from presentation.pages.dcf_setup import render_setup
from presentation.pages.dcf_segments import render_segments
from presentation.pages.dcf_simulation import render_simulation
from presentation.pages.dcf_results import render_results
from presentation.theme.tokens import METRIC_BORDER_COLOR_DCF


# ══════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════

configure_page(
    page_title="SOTP Monte-Carlo DCF",
    page_icon="📊",
)
inject_global_styles(metric_border_color=METRIC_BORDER_COLOR_DCF)


# ══════════════════════════════════════════════════════════════════════════
# Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════

def _collect_config() -> dict:
    return collect_config(dict(st.session_state))


# Keys that belong to widgets – must never be touched by _apply_config.
_WIDGET_KEYS: frozenset[str] = frozenset({
    "config_upload",
})


def _is_config_key(k: str) -> bool:
    """Return True if *k* is a config-managed session-state key."""
    return (
        k.startswith(("setup_", "bridge_", "seg_", "corr_"))
        or k.startswith("wizard_")
        or (len(k) > 2 and k[1].isdigit() and k.startswith("s") and "_" in k)
        or k in ("results", "config", "_config_just_loaded")
    )


def _apply_config(cfg: dict) -> None:
    updated = apply_config(cfg, dict(st.session_state))
    # Only delete/set config-managed keys. Widget-bound keys (file uploaders
    # etc.) cannot be modified after the widget is instantiated and must be
    # left alone to avoid StreamlitAPIException.
    for k in list(st.session_state.keys()):
        if k in _WIDGET_KEYS or not _is_config_key(k):
            continue
        if k not in updated:
            del st.session_state[k]
    for k, v in updated.items():
        if k in _WIDGET_KEYS or not _is_config_key(k):
            continue
        st.session_state[k] = v
    st.session_state["wizard_step"] = "setup"
    st.session_state["wizard_setup"] = None
    st.session_state["wizard_segments"] = None
    st.session_state["_config_just_loaded"] = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Title & sidebar
# ══════════════════════════════════════════════════════════════════════════

render_app_header(
    title="📊 Sum-of-the-Parts Monte-Carlo DCF Modell",
    caption=(
        "Stochastische Unternehmensbewertung · FCFF-Ansatz · "
        "Vektorisierte Simulation"
    ),
)

with st.sidebar:
    st.header("SOTP MC-DCF")
    st.markdown("---")
    if "results" in st.session_state and st.session_state.results is not None:
        r = st.session_state.results
        st.success("Simulation abgeschlossen")
        st.metric("Ø Enterprise Value", f"{np.mean(r.total_ev):,.1f}")
        st.metric("Ø Equity Value", f"{np.mean(r.equity_values):,.1f}")
        st.metric("Ø Preis/Aktie", f"{np.mean(r.price_per_share):,.2f}")
        st.metric("Iterationen", f"{r.n_simulations:,}")
    else:
        st.info("Noch keine Simulation durchgeführt.")

    st.markdown("---")
    st.subheader("💾 Speichern / Laden")

    save_name = st.text_input(
        "Modellname", value="SOTP_Modell",
        help="Dateiname für den JSON-Export.",
    )
    cfg_json = json.dumps(_collect_config(), indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Konfiguration speichern",
        data=cfg_json,
        file_name=f"{save_name}_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("")
    uploaded_cfg = st.file_uploader(
        "Konfiguration laden (.json)", type=["json"],
    )
    if uploaded_cfg is not None:
        if st.button("⬆️ Konfiguration anwenden", use_container_width=True, type="primary"):
            try:
                raw = uploaded_cfg.getvalue().decode("utf-8")
                loaded = json.loads(raw)
                if "ui_state" not in loaded and "setup" not in loaded:
                    st.error("Ungültiges Dateiformat – erwartete 'ui_state' oder Legacy-'setup'.")
                else:
                    _apply_config(loaded)
            except json.JSONDecodeError:
                st.error("Ungültige JSON-Datei.")
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")

    if st.session_state.pop("_config_just_loaded", False):
        st.success("✅ Konfiguration erfolgreich geladen!")

    render_sidebar_footer(tech_stack="Built with Streamlit · NumPy · Plotly")


# ══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "config" not in st.session_state:
    st.session_state.config = None
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = "setup"
if "wizard_setup" not in st.session_state:
    st.session_state.wizard_setup = None
if "wizard_segments" not in st.session_state:
    st.session_state.wizard_segments = None


# ══════════════════════════════════════════════════════════════════════════
# Wizard flow – delegates to presentation.pages.dcf_*
# ══════════════════════════════════════════════════════════════════════════

WIZARD_STEPS: list[tuple[str, str]] = [
    ("setup", "⚙️ Setup"),
    ("segments", "🏢 Segmente"),
    ("simulation", "🎲 Simulation"),
    ("results", "📈 Ergebnisse"),
]


def _step_index(step: str) -> int:
    for idx, (step_id, _) in enumerate(WIZARD_STEPS):
        if step_id == step:
            return idx
    return 0


def _go_to(step: str) -> None:
    st.session_state.wizard_step = step
    st.rerun()


current_step = st.session_state.wizard_step
current_idx = _step_index(current_step)

st.markdown("### Schritt-für-Schritt Workflow")
progress_cols = st.columns(len(WIZARD_STEPS))
for idx, ((step_id, label), col) in enumerate(zip(WIZARD_STEPS, progress_cols)):
    if idx < current_idx:
        state_label = "✅"
    elif idx == current_idx:
        state_label = "➡️"
    else:
        state_label = "◻️"
    col.caption(f"{state_label} {label}")

st.progress((current_idx + 1) / len(WIZARD_STEPS))
st.markdown("")

container = st.container()

if current_step == "setup":
    setup = render_setup(container)

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        st.button("Zurück", disabled=True, use_container_width=True)
    with nav_right:
        if st.button("Weiter: Segmente", type="primary", use_container_width=True):
            previous_setup = st.session_state.wizard_setup or {}
            previous_n_segments = previous_setup.get("n_segments")
            st.session_state.wizard_setup = setup
            if previous_n_segments != setup.get("n_segments"):
                st.session_state.wizard_segments = None
                st.session_state.results = None
                st.session_state.config = None
            _go_to("segments")

elif current_step == "segments":
    setup = st.session_state.wizard_setup
    if setup is None:
        st.warning("Bitte zuerst den Setup-Schritt ausfüllen.")
        if st.button("Zurück zu Setup", type="primary"):
            _go_to("setup")
    else:
        segment_configs = render_segments(container, int(setup["n_segments"]))
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            if st.button("Zurück: Setup", use_container_width=True):
                _go_to("setup")
        with nav_right:
            if st.button("Weiter: Simulation", type="primary", use_container_width=True):
                st.session_state.wizard_segments = segment_configs
                st.session_state.results = None
                st.session_state.config = None
                _go_to("simulation")

elif current_step == "simulation":
    setup = st.session_state.wizard_setup
    segment_configs = st.session_state.wizard_segments

    if setup is None:
        st.warning("Bitte zuerst den Setup-Schritt ausfüllen.")
        if st.button("Zurück zu Setup", type="primary"):
            _go_to("setup")
    elif not segment_configs:
        st.warning("Bitte zuerst die Segmente konfigurieren.")
        if st.button("Zurück zu Segmente", type="primary"):
            _go_to("segments")
    else:
        render_simulation(container, setup, segment_configs)
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            if st.button("Zurück: Segmente", use_container_width=True):
                _go_to("segments")
        with nav_right:
            can_go_results = st.session_state.results is not None
            if st.button(
                "Weiter: Ergebnisse",
                type="primary",
                use_container_width=True,
                disabled=not can_go_results,
            ):
                _go_to("results")

elif current_step == "results":
    render_results(container)
    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Zurück: Simulation", use_container_width=True):
            _go_to("simulation")
    with nav_right:
        st.button("Weiter", disabled=True, use_container_width=True)
else:
    st.session_state.wizard_step = "setup"
    st.rerun()
