"""
SOTP Monte-Carlo DCF Simulation – Streamlit Application
========================================================

Main entry point.  Run with::

    streamlit run app.py

The application uses four tabs, each implemented in a dedicated module
under ``presentation/pages/``:

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
from presentation.pages.dcf_setup import render_setup
from presentation.pages.dcf_segments import render_segments
from presentation.pages.dcf_simulation import render_simulation
from presentation.pages.dcf_results import render_results


# ══════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SOTP Monte-Carlo DCF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 500; }
    details summary { font-weight: 600; }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════

def _collect_config() -> dict:
    return collect_config(dict(st.session_state))


def _apply_config(cfg: dict) -> None:
    updated = apply_config(cfg, dict(st.session_state))
    for k in list(st.session_state.keys()):
        if k not in updated:
            del st.session_state[k]
    for k, v in updated.items():
        st.session_state[k] = v
    st.session_state["_config_just_loaded"] = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Title & sidebar
# ══════════════════════════════════════════════════════════════════════════

st.title("📊 Sum-of-the-Parts Monte-Carlo DCF Modell")
st.caption(
    "Stochastische Unternehmensbewertung · FCFF-Ansatz · "
    "Vektorisierte Simulation"
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
        "Modellname", value="SOTP_Modell", key="save_model_name",
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
        "Konfiguration laden (.json)", type=["json"], key="config_upload",
    )
    if uploaded_cfg is not None:
        if st.button("⬆️ Konfiguration anwenden", use_container_width=True, type="primary"):
            try:
                raw = uploaded_cfg.getvalue().decode("utf-8")
                loaded = json.loads(raw)
                if "setup" not in loaded:
                    st.error("Ungültiges Dateiformat – 'setup' fehlt.")
                else:
                    _apply_config(loaded)
            except json.JSONDecodeError:
                st.error("Ungültige JSON-Datei.")
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")

    if st.session_state.pop("_config_just_loaded", False):
        st.success("✅ Konfiguration erfolgreich geladen!")

    st.markdown("---")
    st.caption("Built with Streamlit · NumPy · Plotly")


# ══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "config" not in st.session_state:
    st.session_state.config = None


# ══════════════════════════════════════════════════════════════════════════
# Tabs – delegates to presentation.pages.dcf_*
# ══════════════════════════════════════════════════════════════════════════

tab_setup, tab_segments, tab_sim, tab_results = st.tabs([
    "⚙️ Setup",
    "🏢 Segmente",
    "🎲 Simulation",
    "📈 Ergebnisse",
])

setup = render_setup(tab_setup)
segment_configs = render_segments(tab_segments, setup["n_segments"])
render_simulation(tab_sim, setup, segment_configs)
render_results(tab_results)
