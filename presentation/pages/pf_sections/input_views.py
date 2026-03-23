"""Black-Litterman views section for portfolio setup."""
from __future__ import annotations

import streamlit as st


def render_input_views_section(asset_configs: list[dict]) -> tuple[bool, list[dict]]:
    """Render optional BL views and return toggle plus serialized views."""
    st.divider()
    st.subheader("🧭 Black-Litterman Views (optional)")

    enable_bl = st.checkbox(
        "Black-Litterman Modell aktivieren",
        value=False,
        key="bl_enable",
        help="Kombination aus Marktgleichgewicht und subjektiven "
             "Analysteneinschätzungen für robustere Renditeerwartungen.",
    )

    bl_views: list[dict] = []
    n_total = len(asset_configs)
    if enable_bl:
        with st.expander("ℹ️ Was ist Black-Litterman?", expanded=False):
            st.markdown(r"""
Das **Black-Litterman-Modell** kombiniert ein Markt-Gleichgewicht (Prior)
mit subjektiven Analystenmeinungen (Views):

$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\right]^{-1}
\left[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}q\right]$$

- **π** = Gleichgewichtsrenditen (aus CAPM)
- **P, q** = Ihre Views (»Aktie X liefert r %«)
- **Ω** = Unsicherheit Ihrer Views (gesteuert durch Konfidenz)
- **τ** = Prior-Unsicherheit (Standard: 0,05)

**Vorteil:** Stabilere Gewichtungen als reines Mean-Variance,
weil Schätzfehler in den Renditeerwartungen reduziert werden.
""")

        n_views = st.number_input(
            "Anzahl Views", min_value=1, max_value=min(n_total, 10),
            value=min(n_total, 1), key="bl_n_views",
        )
        asset_names = [ac["name"] for ac in asset_configs]
        for vi in range(int(n_views)):
            vc1, vc2, vc3 = st.columns([2, 1, 1])
            bl_asset = vc1.selectbox(
                f"Asset (View {vi+1})", asset_names,
                key=f"bl_asset_{vi}",
            )
            bl_ret = vc2.number_input(
                f"Erw. Rendite (%) View {vi+1}",
                value=10.0, min_value=-100.0, max_value=500.0,
                format="%.1f", key=f"bl_ret_{vi}",
            )
            bl_conf = vc3.slider(
                f"Konfidenz View {vi+1}",
                min_value=0.1, max_value=0.99, value=0.5,
                step=0.05, key=f"bl_conf_{vi}",
            )
            bl_views.append({
                "asset_name": bl_asset,
                "expected_return": bl_ret / 100.0,
                "confidence": bl_conf,
            })

    return enable_bl, bl_views
