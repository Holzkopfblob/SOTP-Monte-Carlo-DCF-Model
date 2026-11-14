"""
Treffsicherheit – wie gut kalibriert sind die Bewertungen?
==========================================================
Liest realisierungen.csv und das Annahmen-Register (beide neben den Bewertungen, außerhalb des Repos) und
vergleicht realisierte Werte mit den Jahr-1-Verteilungen der damaligen Bewertung (PIT).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from application.calibration_service import convergence, evaluate, load_realisierungen
from application.portfolio_service import load_depot
from domain.calibration import summarize, verdict
from infrastructure.market_data import fetch_closes
from presentation.charts.calibration import interval_chart, pit_histogram


@st.cache_data(show_spinner=False)
def _evaluate(csv_path: str, mtime: float, folder: str) -> pd.DataFrame:  # mtime nur als Cache-Schlüssel
    return evaluate(load_realisierungen(csv_path), folder)


@st.cache_data(ttl=3600, show_spinner=False)
def _prices(tickers: tuple[str, ...]) -> dict[str, float]:
    closes = fetch_closes(list(tickers))
    return {t: float(closes[t].dropna().iloc[-1]) for t in tickers}


def render_calibration(container) -> None:
    with container:
        st.markdown("## 🎯 Treffsicherheit")
        st.caption("Jede Bewertung ist eine Wahrscheinlichkeitsaussage. Hier wird geprüft, ob „90 % sicher“ "
                   "auch 90 % bedeutet: Quantil-Rang (PIT) des später realisierten Werts in der damaligen Verteilung.")
        depot = load_depot(os.environ.get("SOTP_DEPOT", "depot.json"))
        folder = Path(depot.get("bewertungen_ordner", ""))
        real_csv = folder / "realisierungen.csv"
        if not real_csv.exists():
            st.info("Noch keine realisierten Werte (realisierungen.csv neben den Bewertungen).")
            return
        with st.spinner("Bewertungen neu simulieren …"):
            ev = _evaluate(str(real_csv), real_csv.stat().st_mtime, str(folder))
        ok = ev[ev["fehler"] == ""].copy()
        bad = ev[ev["fehler"] != ""]
        s = summarize(ok["pit"].tolist())

        k = st.columns(4)
        k[0].metric("Prüfpunkte", s.get("n", 0), border=True)
        k[1].metric("Im P5–P95-Band", f"{s.get('im_90_band', 0):.0%}".replace(".", ","), "ideal 90 %",
                    delta_color="off", border=True)
        k[2].metric("Mittlerer Rang", f"{s.get('mittel', 0):.2f}".replace(".", ","), "ideal 0,50",
                    delta_color="off", border=True)
        k[3].metric("Lesart", verdict(s), border=True)
        if len(ok):
            half = (ok["p95"] - ok["p5"]) / 2
            err = (ok["ist_pct"].astype(float) - ok["p50"]).abs()
            factor = float(np.quantile(err / half.replace(0, np.nan), 0.9))
            de = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731 – nur Zahlen, nicht den Satz umformatieren
            st.info(f"Typischer Fehler {de(err.mean())} pp bei einer Modell-Bandbreite (P5–P95) von "
                    f"{de((2 * half).mean())} pp. Für 90 % Abdeckung müssten die Bänder etwa **{de(factor)}×** "
                    f"breiter sein (bei n = {len(ok)} nur Tendenz).", icon="📏")

        t1, t2, t3 = st.tabs(["📊 Prognose gegen Ist", "📈 Rang-Verteilung", "🧭 Konvergenz Kurs → Wert"])
        with t1:
            if len(ok):
                st.plotly_chart(interval_chart(ok), width="stretch", config={"displayModeBar": False})
            st.dataframe(ok[["firma", "feld", "zeitraum", "p5", "p50", "p95", "ist_pct", "pit", "quelle", "pruefstatus",
                             "notiz"]], hide_index=True, width="stretch",
                         column_config={"p5": st.column_config.NumberColumn("P5 %", format="%.1f"),
                                        "p50": st.column_config.NumberColumn("P50 %", format="%.1f"),
                                        "p95": st.column_config.NumberColumn("P95 %", format="%.1f"),
                                        "ist_pct": st.column_config.NumberColumn("Ist %"),
                                        "pit": st.column_config.ProgressColumn("Rang", min_value=0.0, max_value=1.0,
                                                                               format="%.2f")})
            if len(bad):
                st.caption("Nicht auswertbar: " + "; ".join(f"{a} ({b})" for a, b in zip(bad["firma"], bad["fehler"])))
        with t2:
            if len(ok):
                st.plotly_chart(pit_histogram(ok["pit"].tolist()), width="stretch", config={"displayModeBar": False})
            st.caption("Gleichverteilt = gut kalibriert. Häufung links ⇒ Ist liegt unter dem Modell (zu optimistisch), "
                       "rechts ⇒ zu pessimistisch, an beiden Rändern ⇒ Bänder zu eng, in der Mitte ⇒ zu breit.")
        with t3:
            tick = {p["firma"]: p["ticker"] for p in depot.get("positionen", [])}
            register = folder / "annahmen_register.csv"
            if not register.exists() or not tick:
                st.info("Register oder Depot-Ticker fehlen.")
                return
            try:
                px = _prices(tuple(sorted(set(tick.values()))))
            except Exception as e:  # offline
                st.warning(f"Kurse nicht abrufbar: {e}")
                return
            conv = convergence(register, {f: px[t] for f, t in tick.items() if t in px})
            st.dataframe(conv.rename(columns={"datum": "Stichtag", "kurs": "Kurs damals", "p50": "Wert P50",
                                              "kurs_heute": "Kurs heute", "luecke_damals": "Lücke damals",
                                              "luecke_heute": "Lücke heute", "geschlossen": "Geschlossen"}),
                         hide_index=True, width="stretch",
                         column_config={"Lücke damals": st.column_config.NumberColumn(format="%.2f"),
                                        "Lücke heute": st.column_config.NumberColumn(format="%.2f"),
                                        "Geschlossen": st.column_config.NumberColumn(format="percent")})
            st.caption("Lücke = ln(Wert P50 / Kurs). „Geschlossen“ = Anteil der damaligen Lücke, den der Kurs seither "
                       "aufgeholt hat (negativ = Lücke größer geworden). Kurse in Bewertungswährung; aussagekräftig erst "
                       "nach Monaten – daraus lassen sich später Horizont und „Markt hat recht“ (p) je Gruppe schätzen.")
