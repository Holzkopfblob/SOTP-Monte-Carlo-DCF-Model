"""
Basisraten (Außensicht, v3-Plan Phase 2): Wie haben sich Firmen mit ähnlichem Startpunkt tatsächlich entwickelt?
Methode wie Mauboussin „Base Rates“: Ergebnisverteilungen (Umsatz-CAGR, Margenänderung) über h Jahre,
bedingt auf Branche, Größe, bisheriges Wachstum und Startmarge. Zu wenige Fälle ⇒ Bedingungen schrittweise lockern.

Eingabe ist ein Jahres-Panel (cik, jahr, umsatz, ebit, da, sic), z. B. aus infrastructure/sec_xbrl.load_panel.
# ponytail: Überlebensverzerrung – Firmen, die verschwinden, fehlen im h-Jahres-Ergebnis ⇒ Wachstum leicht zu
# optimistisch, untere Ränder zu dünn. Upgrade: Delistings als eigener Ausgang (z. B. −100 %/Übernahme getrennt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEKTOREN = [  # (SIC von, bis, Name) – erste passende Zeile gewinnt
    (4810, 4899, "Telko"), (4900, 4949, "Versorger"), (1300, 1399, "Energie"), (2900, 2999, "Energie"),
    (6000, 6799, "Finanz"), (7370, 7379, "Software/IT"), (3670, 3679, "Halbleiter/Elektronik"),
    (3570, 3579, "Halbleiter/Elektronik"), (3600, 3699, "Halbleiter/Elektronik"),
    (2830, 2836, "Gesundheit"), (3840, 3851, "Gesundheit"), (8000, 8099, "Gesundheit"),
    (3710, 3716, "Auto"), (1000, 1499, "Rohstoffe/Chemie"), (2600, 2899, "Rohstoffe/Chemie"),
    (3400, 3799, "Industrie"), (1500, 1799, "Industrie"), (4000, 4799, "Industrie"),
    (2000, 2399, "Konsum"), (5000, 5999, "Konsum"), (7000, 7369, "Konsum/Dienste"), (7380, 8999, "Konsum/Dienste"),
]
G_EDGES = [-np.inf, 0.0, 0.05, 0.10, 0.20, 0.40, np.inf]   # bisheriges Wachstum (1 J.)
M_EDGES = [-np.inf, 0.10, 0.20, 0.30, 0.45, np.inf]        # EBITDA-Marge
S_EDGES = [0, 1e9, 1e10, np.inf]                           # Umsatz in USD
MIN_UMSATZ = 50e6                                          # Kleinstfirmen verrauschen die Verteilungen
LOCKERUNG = ["g_start", "m_start", "groesse", "sektor"]   # Reihenfolge beim Lockern


def sektor(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Sonstige"
    return next((name for lo, hi, name in SEKTOREN if lo <= s <= hi), "Sonstige")


def bucket(x, edges) -> np.ndarray:
    return np.digitize(np.asarray(x, float), edges[1:-1])


def outcomes(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Je (Firma, Startjahr t): Startmerkmale und Ergebnis über h Jahre (cagr, dm = Margenänderung in Punkten)."""
    p = panel[["cik", "jahr", "umsatz", "ebit", "da", "sic"]].copy()
    p["marge"] = (p.ebit + p.da) / p.umsatz
    prev = p.assign(jahr=p.jahr + 1)[["cik", "jahr", "umsatz"]].rename(columns={"umsatz": "umsatz_vj"})
    fut = p.assign(jahr=p.jahr - h)[["cik", "jahr", "umsatz", "marge"]].rename(
        columns={"umsatz": "umsatz_h", "marge": "marge_h"})
    o = p.merge(prev, on=["cik", "jahr"], how="left").merge(fut, on=["cik", "jahr"], how="inner")
    o = o[(o.umsatz >= MIN_UMSATZ) & (o.umsatz_h > 0)]
    o["cagr"] = (o.umsatz_h / o.umsatz) ** (1 / h) - 1
    o["dm"] = o.marge_h - o.marge
    o["g_vj"] = o.umsatz / o.umsatz_vj - 1
    o["sektor"] = o.sic.map(sektor)
    o["groesse"] = bucket(o.umsatz, S_EDGES)
    o["g_start"] = np.where(o.g_vj.notna(), bucket(o.g_vj, G_EDGES), -1)
    o["m_start"] = np.where(o.marge.notna(), bucket(o.marge, M_EDGES), -1)
    return o[o.sektor != "Finanz"]       # EBITDA/Umsatz ist bei Banken/Versicherern nicht sinnvoll


def conditional(o: pd.DataFrame, wert: str, min_n: int = 200, lockerung=LOCKERUNG, **bed) -> tuple[np.ndarray, dict]:
    """Ergebnisse *wert* ('cagr' oder 'dm') für die Bedingungen *bed* (sektor, groesse, g_start, m_start).
    Bei < min_n Fällen wird in der Reihenfolge *lockerung* gelockert. Rückgabe: Werte, genutzte Bedingungen."""
    bed = {k: v for k, v in bed.items() if v is not None}
    for drop in [None, *lockerung]:
        if drop:
            bed.pop(drop, None)
        m = np.ones(len(o), bool)
        for k, v in bed.items():
            m &= (o[k] == v).to_numpy()
        vals = o.loc[m, wert].dropna().to_numpy()
        if len(vals) >= min_n:
            return vals, dict(bed)
    return o[wert].dropna().to_numpy(), {}


def percentile_of(vals: np.ndarray, x: float) -> float:
    """Anteil der Basisrate unter x (0…1)."""
    return float(np.mean(np.asarray(vals) < x))


def quantiles(vals: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    return {f"p{q}": float(np.percentile(vals, q)) for q in qs} | {"n": int(len(vals))}
