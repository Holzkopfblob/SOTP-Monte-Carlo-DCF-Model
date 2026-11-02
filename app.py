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
import re
from datetime import datetime

import numpy as np
import streamlit as st

from infrastructure.config_io import apply_config, collect_config, parse_config_text
from presentation.layout.base import (
    configure_page,
    inject_global_styles,
    render_app_header,
    render_sidebar_footer,
)
from presentation.pages.dcf_compare import render_compare
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

_SEGMENT_DIST_KEY_PATTERN = re.compile(r"^s\d+_")


def _is_config_key(k: str) -> bool:
    """Return True if *k* is a config-managed session-state key."""
    return (
        k.startswith(("setup_", "bridge_", "seg_", "corr_"))
        or k.startswith("cfg_")
        or k.startswith("wizard_")
        or (len(k) > 2 and k[1].isdigit() and k.startswith("s") and "_" in k)
        or k in ("results", "config", "_config_just_loaded")
    )


def _is_config_widget_state_key(k: str) -> bool:
    """Return True for actual widget-bound config keys.

    These keys can be removed by Streamlit when their widgets are not rendered
    in the current wizard step (e.g. segment fields while viewing setup).
    """
    return (
        k.startswith(("setup_", "bridge_", "seg_", "corr_"))
        or _SEGMENT_DIST_KEY_PATTERN.match(k) is not None
    )


def _preserve_config_widget_state() -> None:
    """Detach config widget keys from Streamlit's widget lifecycle cleanup."""
    for key in list(st.session_state.keys()):
        if _is_config_widget_state_key(key):
            st.session_state[key] = st.session_state[key]


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


# Keep imported/setup segment widget state alive across wizard step switches.
_preserve_config_widget_state()


# ══════════════════════════════════════════════════════════════════════════
# Title & sidebar
# ══════════════════════════════════════════════════════════════════════════

render_app_header(
    title="📊 SOTP Monte-Carlo DCF",
    caption=(
        "Stochastische Unternehmensbewertung · FCFF-Ansatz · "
        "Vektorisierte Simulation"
    ),
)

with st.sidebar:
    st.header("SOTP MC-DCF")
    view = st.radio(
        "Ansicht", ["🧮 Bewertung", "🔍 Versionsvergleich"],
        key="app_view", horizontal=True, label_visibility="collapsed",
    )
    meta = (st.session_state.get("cfg_passthrough") or {}).get("metadata") or {}
    if meta.get("company"):
        st.caption(
            f"**{meta['company']}**"
            + (f" · Stand {meta['stichtag']}" if meta.get("stichtag") else "")
            + (f" · {meta['report_currency']}" if meta.get("report_currency") else "")
        )

    # Research import first – it is where every valuation starts.
    with st.expander("📋 Research-Ergebnis einfügen", expanded=not meta):
        pasted = st.text_area(
            "LLM-Antwort", key="config_paste", height=160,
            label_visibility="collapsed",
            placeholder="Komplette Antwort des Research-Prompts oder nur den JSON-Block einfügen …",
        )
        if st.button("Übernehmen", width="stretch", type="primary", disabled=not pasted.strip()):
            try:
                _apply_config(parse_config_text(pasted))
            except ValueError as e:
                st.error(str(e))
    if st.session_state.pop("_config_just_loaded", False):
        st.success("✅ Konfiguration erfolgreich geladen!")

    st.divider()
    if st.session_state.get("results") is not None:
        r = st.session_state.results
        p5, p50, p95 = np.percentile(r.price_per_share, [5, 50, 95])
        st.success("Simulation abgeschlossen")
        st.caption(
            f"Fairer Wert je Aktie (Median) **{p50:,.2f}**  \n"
            f"P5 – P95: {p5:,.0f} – {p95:,.0f} · {r.n_simulations:,} Szenarien"
        )
    else:
        st.caption("Noch keine Simulation durchgeführt.")

    st.divider()
    st.subheader("💾 Speichern / Laden")
    save_name = st.text_input(
        "Modellname", value=str(meta.get("company") or "SOTP_Modell").replace(" ", "_"),
        help="Dateiname für den JSON-Export. Mit Simulationsergebnis gespeichert "
             "wird die Datei zum vergleichbaren Zeitpunkt im 🔍 Versionsvergleich.",
    )
    cfg_json = json.dumps(_collect_config(), indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Konfiguration speichern",
        data=cfg_json,
        file_name=f"{save_name}_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        width="stretch",
    )
    uploaded_cfg = st.file_uploader(
        "Konfiguration laden (.json)", type=["json"],
    )
    if uploaded_cfg is not None:
        if st.button("⬆️ Konfiguration anwenden", width="stretch", type="primary"):
            try:
                _apply_config(parse_config_text(uploaded_cfg.getvalue().decode("utf-8")))
            except ValueError as e:
                st.error(str(e))

    verz = (st.session_state.get("cfg_passthrough") or {}).get("quellenverzeichnis")
    if verz:
        with st.expander("📚 Quellenverzeichnis"):
            for ref_key, ref in verz.items():
                st.markdown(f"**[{ref_key}]** {ref}")

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

if view == "🔍 Versionsvergleich":
    render_compare(st.container())
    st.stop()


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

steps_md = []
for idx, (step_id, label) in enumerate(WIZARD_STEPS):
    name = f"{idx + 1} · {label.split(' ', 1)[1]}"
    if idx < current_idx:
        steps_md.append(f":green[✓ {name}]")
    elif idx == current_idx:
        steps_md.append(f":blue-background[**{name}**]")
    else:
        steps_md.append(f":gray[{name}]")
st.markdown("  →  ".join(steps_md))

container = st.container()

if current_step == "setup":
    setup = render_setup(container)

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        st.button("Zurück", disabled=True, width="stretch")
    with nav_right:
        if st.button("Weiter: Segmente", type="primary", width="stretch"):
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
        st.warning("Setup fehlt. Füllen Sie zuerst den Schritt **⚙️ Setup** aus und klicken Sie anschließend auf **Weiter: Segmente**.")
        if st.button("Zurück zu Setup", type="primary"):
            _go_to("setup")
    else:
        segment_configs = render_segments(container, int(setup["n_segments"]))
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            if st.button("Zurück: Setup", width="stretch"):
                _go_to("setup")
        with nav_right:
            if st.button("Weiter: Simulation", type="primary", width="stretch"):
                st.session_state.wizard_segments = segment_configs
                st.session_state.results = None
                st.session_state.config = None
                _go_to("simulation")

elif current_step == "simulation":
    setup = st.session_state.wizard_setup
    segment_configs = st.session_state.wizard_segments

    if setup is None:
        st.warning("Setup fehlt. Gehen Sie zu **⚙️ Setup**, vervollständigen Sie die Eingaben und wechseln Sie dann zurück.")
        if st.button("Zurück zu Setup", type="primary"):
            _go_to("setup")
    elif not segment_configs:
        st.warning("Segmentkonfiguration fehlt. Füllen Sie zuerst den Schritt **🏢 Segmente** vollständig aus.")
        if st.button("Zurück zu Segmente", type="primary"):
            _go_to("segments")
    else:
        render_simulation(container, setup, segment_configs)
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            if st.button("Zurück: Segmente", width="stretch"):
                _go_to("segments")
        with nav_right:
            can_go_results = st.session_state.results is not None
            if st.button(
                "Weiter: Ergebnisse",
                type="primary",
                width="stretch",
                disabled=not can_go_results,
            ):
                _go_to("results")

elif current_step == "results":
    render_results(container)
    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Zurück: Simulation", width="stretch"):
            _go_to("simulation")
    with nav_right:
        st.button("Weiter", disabled=True, width="stretch")
else:
    st.session_state.wizard_step = "setup"
    st.rerun()
