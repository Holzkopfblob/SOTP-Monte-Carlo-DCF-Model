"""Bestandssicht: was im Depot liegt, was es gebracht hat, was das Modell dazu sagt und was als Nächstes ansteht.

Verbindet das Ergebnis von rebalance (Kurse, Bewertung, Alpha, Ziel, Order) mit den Depotdaten (Einstand, Kaufdatum)
und der Research-Ablage im Vault ($SOTP_RESEARCH: termine.csv, Firmen/<TICKER>.md).
"""
from __future__ import annotations

import csv
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from domain.ledger import book

RESEARCH = Path(os.environ.get("SOTP_RESEARCH",
                               r"C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen\SOTP-Research"))
CASH = "Cash (EUR)"


def next_events(research: Path = RESEARCH, today: date | None = None) -> dict[str, str]:
    """Nächster offener Termin je Ticker aus termine.csv als Text (überfällige zuerst markiert)."""
    today = today or date.today()
    path = research / "termine.csv"
    if not path.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r.get("status") == "erledigt":
                continue
            if r["ticker"] not in out or r["datum"] < out[r["ticker"]][0]:
                out[r["ticker"]] = (r["datum"], r["ereignis"])
    return {t: ("⚠ " if d < today.isoformat() else "") + f"{d[8:10]}.{d[5:7]}. {e}" for t, (d, e) in out.items()}


def thesis(ticker: str, research: Path = RESEARCH, max_len: int = 160) -> str:
    """Erster inhaltlicher Satz unter „## These“ im Dossier (ohne Entwurfs-Hinweis), sonst leer."""
    path = research / "Firmen" / f"{ticker}.md"
    if not path.exists():
        return ""
    m = re.search(r"## These\n(.*?)(\n## |\Z)", path.read_text(encoding="utf-8"), re.S)
    if not m:
        return ""
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith("_")]
    text = " ".join(lines)
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
    return first if len(first) <= max_len else first[: max_len - 1] + "…"


def _ledger_cols(k, unrealisiert: float) -> dict:
    """Realisiert, Dividenden, Gebühren und Gesamtergebnis (unrealisiert + realisiert + Dividenden) je Titel."""
    if k is None:
        return {"Realisiert €": np.nan, "Dividenden €": np.nan, "Ergebnis gesamt €": np.nan}
    return {"Realisiert €": k.realisiert, "Dividenden €": k.dividenden,
            "Ergebnis gesamt €": unrealisiert + k.realisiert + k.dividenden}


def _action(row: pd.Series) -> str:
    if row.get("Umsetzung") == "Sparplan":
        return f"Sparplan {row['Sparplan €/Monat']:,.0f} €/Monat".replace(",", ".")
    o = float(row.get("Order €") or 0.0)
    if o > 0.5:
        return f"Kaufen {o:,.0f} €".replace(",", ".")
    if o < -0.5:
        return ("Alles verkaufen" if row["Ziel-Gewicht"] < 1e-4 else f"Verkaufen {-o:,.0f} €").replace(",", ".")
    return "Halten"


def closed_positions(positions: list[dict], transaktionen: list[dict]) -> pd.DataFrame:
    """Vollständig verkaufte Positionen mit Ergebnis (realisiert + Dividenden − sonstige Gebühren)."""
    names = {p["ticker"]: p["firma"] for p in positions}
    rows = []
    for tic, k in book(transaktionen or []).items():
        if not k.geschlossen:
            continue
        tx = [t for t in transaktionen if t["ticker"] == tic]
        rows.append({"Firma": names.get(tic, tic), "Ticker": tic,
                     "Gekauft": min(str(t["datum"]) for t in tx if t["typ"] == "kauf"),
                     "Verkauft": max(str(t["datum"]) for t in tx if t["typ"] == "verkauf"),
                     "Investiert €": k.investiert, "Erlös €": k.erloese,
                     "Realisiert €": k.realisiert, "Dividenden €": k.dividenden,
                     "Ergebnis €": k.realisiert + k.dividenden,
                     "Ergebnis %": (k.realisiert + k.dividenden) / k.investiert if k.investiert else np.nan})
    return pd.DataFrame(rows)


def holdings(table: pd.DataFrame, positions: list[dict], cash_eur: float, today: date | None = None,
             research: Path = RESEARCH, transaktionen: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(gehaltene Positionen, neue Kandidaten mit Ziel > 0, Summen). Gewinn = (Kurs − Einstand) × Stück;
    mit Transaktionsbuch zusätzlich realisierte Gewinne (FIFO), Dividenden, Gebühren und Gesamtergebnis je Titel."""
    today = today or date.today()
    konten = book(transaktionen or [])
    t = table[table["Firma"] != CASH].set_index("Firma")
    events = next_events(research, today)
    rows = []
    for pos in positions:
        if pos["firma"] not in t.index:
            continue
        r = t.loc[pos["firma"]]
        st_ = float(pos.get("stueck") or 0.0)
        ein = pos.get("einstand_eur")
        ein = float(ein) if ein not in (None, "") and np.isfinite(float(ein)) else np.nan
        try:
            held_days = (today - date.fromisoformat(pos.get("kaufdatum") or "")).days
        except ValueError:
            held_days = np.nan
        wert = st_ * float(r["Kurs €"])
        rows.append({
            "Firma": pos["firma"], "Ticker": pos["ticker"], "Stück": st_, "Kurs €": float(r["Kurs €"]),
            "Wert €": wert, "Gewicht": float(r["Ist-Gewicht"]),
            "Einstand €": ein, "Gewinn €": (float(r["Kurs €"]) - ein) * st_ if st_ else np.nan,
            "Gewinn %": float(r["Kurs €"]) / ein - 1 if st_ and ein and np.isfinite(ein) else np.nan,
            "Haltedauer (Tage)": held_days,
            **_ledger_cols(konten.get(pos["ticker"]), (float(r["Kurs €"]) - ein) * st_ if st_ and np.isfinite(ein) else 0.0),
            "Fairer Wert €": float(r["Wert P50 €"]), "Potenzial": float(r["Upside P50"]),
            "Rendite p. a.": r.get("Rendite p. a.", np.nan), "Beta": r.get("Beta", np.nan),
            "Alpha p. a.": r.get("Vorsprung p. a.", np.nan),
            "Ziel-Gewicht": float(r["Ziel-Gewicht"]), "Aktion": _action(r),
            "Bewertung": str(r.get("Stand") or ""), "Nächster Termin": events.get(pos["ticker"], ""),
            "These": thesis(pos["ticker"], research),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df, {"wert": 0.0, "cash": cash_eur, "gesamt": cash_eur}
    held = df[df["Stück"] > 0].sort_values("Wert €", ascending=False).reset_index(drop=True)
    new = df[(df["Stück"] <= 0) & (df["Aktion"] != "Halten")].sort_values("Ziel-Gewicht", ascending=False)   # Kauf/Sparplan
    known = held["Einstand €"].notna()
    einstand = float((held.loc[known, "Einstand €"] * held.loc[known, "Stück"]).sum())
    gewinn = float(held.loc[known, "Gewinn €"].sum())
    wert = float(held["Wert €"].sum())
    realisiert = float(sum(k.realisiert for k in konten.values()))
    dividenden = float(sum(k.dividenden for k in konten.values()))
    gebuehren = float(sum(k.gebuehren for k in konten.values()))
    totals = {"wert": wert, "cash": cash_eur, "gesamt": wert + cash_eur, "einstand": einstand, "gewinn": gewinn,
              "realisiert": realisiert, "dividenden": dividenden, "gebuehren": gebuehren,
              "ergebnis": gewinn + realisiert + dividenden,
              "warnungen": [f"{t}: {w}" for t, k in konten.items() for w in k.warnungen],
              "gewinn_pct": gewinn / einstand if einstand else np.nan, "n": int(len(held)),
              "ohne_einstand": held.loc[~known, "Firma"].tolist(),
              "alpha_gewichtet": float(np.nansum(held["Alpha p. a."] * held["Wert €"]) / wert) if wert else np.nan}
    return held, new.reset_index(drop=True), totals
