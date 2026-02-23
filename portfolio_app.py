"""
Portfolio-Strukturierung & Optimierung – Streamlit Application
===============================================================

Thin orchestrator – all heavy UI logic lives in
``presentation.pages.pf_*`` modules.

Starten mit::

    streamlit run portfolio_app.py --server.port 8502
"""
from __future__ import annotations

import streamlit as st

from presentation.pages.pf_input import render_input
from presentation.pages.pf_single import render_single
from presentation.pages.pf_portfolio import render_portfolio
from presentation.pages.pf_frontier import render_frontier
from presentation.pages.pf_stress import render_stress


# ══════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio-Optimierung",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #9467bd;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 500; }
    details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════

st.title("💼 Portfolio-Strukturierung & Optimierung")
st.caption(
    "Statistische Analyse · Markowitz · Kelly · Risk Parity · "
    "Min CVaR · Max Diversifikation · Monte-Carlo Fair-Value-Verteilungen"
)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("💼 Portfolio-Tool")
    st.markdown("---")
    st.markdown(
        "Geben Sie für jede Aktie Ihre Bewertung ein und erhalten Sie "
        "eine optimale Portfoliostruktur mit **7 Methoden** im Vergleich."
    )
    st.markdown("---")

    st.subheader("⚙️ Globale Einstellungen")
    n_mc_sim = st.number_input(
        "MC-Simulationen je Asset",
        min_value=5_000, max_value=200_000, value=50_000, step=5_000,
        help="Mehr Samples → genauere Verteilungen, aber längere Berechnung.",
    )
    global_seed = st.number_input("Random Seed", value=42, min_value=0)
    risk_free_pct = st.number_input(
        "Risikofreier Zins (%)", value=3.0, min_value=0.0, max_value=20.0,
        format="%.2f",
        help="z.B. 10J-Bundesanleihe oder US Treasury Yield.",
    )

    st.markdown("---")

    st.subheader("💾 Konfiguration")
    st.caption("Portfolio-Setup als JSON speichern oder laden.")
    uploaded = st.file_uploader("JSON laden", type=["json"], key="pf_upload")

    st.markdown("---")
    st.caption("Built with Streamlit · NumPy · SciPy · Plotly")


# ══════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════

if "pf_results" not in st.session_state:
    st.session_state.pf_results = None


# ══════════════════════════════════════════════════════════════════════════
# Tabs  →  delegate to page modules
# ══════════════════════════════════════════════════════════════════════════

tab_input, tab_single, tab_portfolio, tab_frontier, tab_stress = st.tabs([
    "📝 Bewertungen eingeben",
    "🔍 Einzeltitel-Analyse",
    "📊 Portfolio-Optimierung",
    "📈 Efficient Frontier",
    "⚡ Stress-Tests",
])

render_input(tab_input, n_mc_sim=n_mc_sim, global_seed=global_seed,
             risk_free_pct=risk_free_pct, uploaded=uploaded)
render_single(tab_single)
render_portfolio(tab_portfolio)
render_frontier(tab_frontier)
render_stress(tab_stress)

