"""
Portfolio – Zielgewichte und Orders aus den SOTP-Bewertungen
=============================================================
Depot (Stück, Cash) wird in einer JSON außerhalb des Repos gespeichert;
Pfad per Umgebungsvariable ``SOTP_DEPOT`` vorbelegt.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from application.portfolio_service import (
    Params, find_valuations, load_depot, rebalance, save_depot, simulate_valuation,
)
from infrastructure.market_data import fetch_closes

_NEWEST = "(neueste)"


@st.cache_data(show_spinner=False)
def _simulate(path: str, mtime: float) -> dict:  # mtime nur als Cache-Schlüssel
    return simulate_valuation(path)


@st.cache_data(ttl=3600, show_spinner=False)
def _closes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_closes(list(tickers))


def render_portfolio(container) -> None:
    with container:
        st.subheader("💼 Portfolio-Balancing")
        st.caption("Fractional Kelly auf den Monte-Carlo-Verteilungen der neuesten Bewertungen. "
                   "Modellwert wird voll geglaubt; Konvergenz Kurs → Wert über den Horizont.")

        path = st.text_input("Depot-Datei", os.environ.get("SOTP_DEPOT", "depot.json"), key="pf_path")
        if st.session_state.get("pf_loaded") != path:
            try:
                st.session_state.pf_depot = load_depot(path)
            except (OSError, ValueError) as e:  # JSONDecodeError ist ein ValueError
                st.error(f"Depot-Datei nicht lesbar: {e}")
                return
            st.session_state.pf_loaded = path
        depot = st.session_state.pf_depot

        folder = st.text_input("Bewertungen-Ordner", depot.get("bewertungen_ordner", ""), key="pf_folder")
        vals = find_valuations(folder) if folder and Path(folder).is_dir() else {}
        if not vals:
            st.warning("Keine Bewertungen im Ordner gefunden.")
            return
        files = {f.name: f for fs in vals.values() for f in fs}

        df = pd.DataFrame(depot.get("positionen") or [], columns=["firma", "ticker", "stueck", "datei", "einstand_eur", "kaufdatum"])
        df["datei"] = df["datei"].fillna("").replace("", _NEWEST)
        edited = st.data_editor(
            df, num_rows="dynamic", width="stretch", key="pf_editor",
            column_config={
                "firma": st.column_config.SelectboxColumn("Firma", options=sorted(vals), required=True),
                "ticker": st.column_config.TextColumn("Ticker (Yahoo)", required=True),
                "stueck": st.column_config.NumberColumn("Stück", min_value=0.0, step=0.000001, format="%.6f", default=0.0),
                "einstand_eur": st.column_config.NumberColumn("Einstand €/Stück", min_value=0.0, format="%.2f"),
                "kaufdatum": st.column_config.TextColumn("Kaufdatum (JJJJ-MM-TT)"),
                "datei": st.column_config.SelectboxColumn(
                    "Bewertung", options=[_NEWEST, *sorted(files)], default=_NEWEST),
            },
        )
        cash = st.number_input("Cash (EUR)", min_value=0.0, value=float(depot.get("cash_eur", 0.0)), step=100.0)

        positions = [
            {"firma": r.firma, "ticker": str(r.ticker).strip(), "stueck": float(r.stueck or 0),
             "datei": "" if r.datei in (None, _NEWEST) else r.datei,
             "einstand_eur": None if pd.isna(r.einstand_eur) else float(r.einstand_eur),
             "kaufdatum": r.kaufdatum if isinstance(r.kaufdatum, str) else ""}
            for r in edited.itertuples() if r.firma and isinstance(r.ticker, str) and r.ticker.strip()
        ]
        if st.button("💾 Depot speichern"):
            save_depot(path, {"bewertungen_ordner": folder, "cash_eur": cash, "positionen": positions})
            st.session_state.pf_depot = load_depot(path)
            st.success(f"Gespeichert: {path}")

        c1, c2, c3 = st.columns(3)
        fraction = c1.slider("Kelly-Faktor", 0.1, 1.0, 0.5, 0.05)
        max_w = c2.slider("Max. Gewicht je Titel", 0.05, 0.5, 0.2, 0.01)
        min_cash = c3.slider("Mindest-Cash (Untergrenze)", 0.0, 0.5, 0.0, 0.01,
                             help="0 = Cash-Quote bestimmt allein der Optimizer.")
        c4, c5, c6 = st.columns(3)
        min_order = c4.number_input("Mindestorder (EUR)", 0.0, value=500.0, step=50.0)
        rf = c5.number_input("Risikofreier Zins p. a.", 0.0, 0.1, 0.025, 0.0025, format="%.4f")
        allow_short = c6.toggle("Short erlauben", False,
                                help="Brutto Σ|Gewicht| ≤ 1 − Mindest-Cash (kein Hebel).")

        if not positions:
            st.info("Positionen eintragen (Firma + Ticker; Stück 0 = Beobachtung).")
            return
        if not st.button("📐 Zielgewichte berechnen", type="primary"):
            return

        with st.spinner("Bewertungen neu simulieren, Kurse laden …"):
            try:
                valuations = []
                for pos in positions:
                    f = files[pos["datei"]] if pos["datei"] else vals[pos["firma"]][-1]
                    valuations.append(_simulate(str(f), f.stat().st_mtime))
                closes = _closes(tuple(sorted({p["ticker"] for p in positions})))
                res = rebalance(positions, valuations, closes, cash,
                                Params(fraction, max_w, min_cash, min_order, allow_short, rf))
            except (ValueError, KeyError) as e:
                st.error(str(e))
                return

        for w in res.warnings:
            st.warning(w)
        pct = {c: "{:.1%}" for c in ["Upside P50", "P(Wert>Kurs)", "Ist-Gewicht", "Ziel-Gewicht", "Δ Gewicht"]}
        st.dataframe(res.table.style.format({**pct, "Kurs €": "{:.2f}", "Wert P50 €": "{:.2f}",
                                             "Order €": "{:,.0f}", "Order Stück": "{:.6g}", "Ziel Stück": "{:.6g}"},
                                            na_rep=""), width="stretch", hide_index=True)
        st.caption("Alle Beträge in EUR, USD zum aktuellen EURUSD-Kurs umgerechnet; „Kurs lokal“ = Börsenkurs in Bewertungswährung.")

        m = st.columns(4)
        m[0].metric("Depotwert heute", f"{res.wealth_ist['start']:,.0f} €")
        for col, q in zip(m[1:], ["p5", "p50", "p95"]):
            col.metric(f"In {Params.horizon_years} J. {q.upper()} (Ziel)", f"{res.wealth_ziel[q]:,.0f} €",
                       f"{res.wealth_ziel[q] - res.wealth_ist[q]:+,.0f} € ggü. Ist")
        with st.expander("Korrelation (Wochenrenditen in EUR, Spearman)"):
            st.dataframe(res.corr.style.format("{:.2f}"), width="stretch")
