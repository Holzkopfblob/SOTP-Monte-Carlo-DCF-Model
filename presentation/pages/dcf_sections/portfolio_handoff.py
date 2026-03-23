"""Portfolio handoff section for DCF results."""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.stats import kurtosis, skew


def render_portfolio_handoff_section(results) -> None:
    """Show distribution parameters for handoff to the portfolio app."""
    st.subheader("🔗 Verteilungsparameter für Portfolio-App")
    st.markdown(
        "Übertragen Sie diese Werte in die **Portfolio-Optimierung** "
        "(`portfolio_app.py`), um die simulierte Fair-Value-Verteilung "
        "dort als Input zu nutzen."
    )

    with st.expander("ℹ️ Wie übertrage ich die Werte?", expanded=False):
        st.markdown("""
### So nutzen Sie die Parameter in der Portfolio-App

1. Notieren Sie sich die **5 Kennzahlen** unten (μ, σ, Schiefe, P5, P95)
2. Öffnen Sie die **Portfolio-App** (`streamlit run portfolio_app.py --server.port 8502`)
3. Wählen Sie als Verteilungstyp **"Aus DCF-App (μ, σ, Schiefe)"**
4. Geben Sie die Werte ein – die App rekonstruiert automatisch die passende Verteilung:
   - **Schiefe ≈ 0** → Normalverteilung (symmetrisch)
   - **Schiefe > 0** → Lognormalverteilung (rechtsschiefe MC-Ergebnisse)
5. Die Portfolio-App generiert daraus eine Fair-Value-Verteilung, die der
   MC-Simulation möglichst nahekommt.

> **Tipp:** Für jedes Unternehmen, das Sie per SOTP-DCF bewertet haben,
> können Sie die Parameter übertragen und so ein **Multi-Aktien-Portfolio**
> optimieren.
""")

    prices = results.price_per_share
    p_mean = float(np.mean(prices))
    p_std = float(np.std(prices))
    p_median = float(np.median(prices))
    p_skew = float(skew(prices))
    p_kurt = float(kurtosis(prices))
    p_p5 = float(np.percentile(prices, 5))
    p_p25 = float(np.percentile(prices, 25))
    p_p75 = float(np.percentile(prices, 75))
    p_p95 = float(np.percentile(prices, 95))

    st.markdown("##### 📋 Parameter zum Übertragen")

    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("μ (Mittelwert)", f"{p_mean:,.2f}")
    pk2.metric("σ (Std.-Abw.)", f"{p_std:,.2f}")
    pk3.metric("Schiefe (Skew)", f"{p_skew:,.3f}")
    pk4.metric("P5", f"{p_p5:,.2f}")
    pk5.metric("P95", f"{p_p95:,.2f}")

    pk6, pk7, pk8, pk9, _ = st.columns(5)
    pk6.metric("Median (P50)", f"{p_median:,.2f}")
    pk7.metric("P25", f"{p_p25:,.2f}")
    pk8.metric("P75", f"{p_p75:,.2f}")
    pk9.metric("Kurtosis", f"{p_kurt:,.3f}")

    if abs(p_skew) < 0.5:
        rec_dist = "Normal"
        st.success(
            f"📊 **Empfehlung: Normalverteilung** (Schiefe = {p_skew:,.3f} ≈ 0) · "
            f"Geben Sie in der Portfolio-App ein: **μ = {p_mean:,.2f}** · **σ = {p_std:,.2f}**"
        )
    else:
        rec_dist = "Lognormal"
        st.info(
            f"📊 **Empfehlung: Lognormalverteilung** (Schiefe = {p_skew:,.3f} ≠ 0) · "
            f"Geben Sie in der Portfolio-App ein: **μ = {p_mean:,.2f}** · **σ = {p_std:,.2f}** · "
            f"**Schiefe = {p_skew:,.3f}**"
        )

    st.markdown("##### 📎 Kopiervorlage")
    st.code(
        f"Verteilungstyp: Aus DCF-App (μ, σ, Schiefe)\n"
        f"μ (Mittelwert):  {p_mean:,.4f}\n"
        f"σ (Std.-Abw.):   {p_std:,.4f}\n"
        f"Schiefe (Skew):  {p_skew:,.4f}\n"
        f"─────────────────────────────\n"
        f"Empf. Verteilung: {rec_dist}\n"
        f"Median:           {p_median:,.4f}\n"
        f"P5 / P95:         {p_p5:,.4f} / {p_p95:,.4f}\n"
        f"Kurtosis:         {p_kurt:,.4f}\n"
        f"Simulationen:     {results.n_simulations:,}",
        language="text",
    )
