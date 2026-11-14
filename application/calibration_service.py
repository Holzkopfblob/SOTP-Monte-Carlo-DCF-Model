"""
Kalibrierungs-Service: Modellverteilungen für Jahr 1 (Wachstum, Marge; Segment und Konzern) aus einer
gespeicherten Bewertung, Abgleich mit realisierten Werten und Konvergenz Kurs → Wert aus dem Register.

realisierungen.csv (liegt neben dem Annahmen-Register, Semikolon-getrennt):
  firma;datei;ebene;segment;feld;ist_pct;zeitraum;quelle;pruefstatus;erfasst_am;notiz
  ebene = konzern | segment, feld = wachstum | marge, ist_pct in Prozent.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from application.simulation_service import SimulationService
from domain.calibration import aggregate, pit, year1_growth
from domain.models import RevenueGrowthMode
from infrastructure.config_io import build_simulation_config

FIELDS = ["firma", "datei", "ebene", "segment", "feld", "ist_pct", "zeitraum", "quelle", "pruefstatus",
          "erfasst_am", "notiz"]


@lru_cache(maxsize=64)
def year1_distributions(path: str) -> dict[str, dict[str, np.ndarray]]:
    """{"Konzern"|Segmentname: {"wachstum": Samples, "marge": Samples}} in Dezimal, Jahr 1."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    sc = build_simulation_config(cfg)
    r = SimulationService.run_simulation(sc)
    out, base, gs, ms = {}, [], [], []
    for seg in sc.segments:
        s = r.input_samples
        g0 = s[f"{seg.name} | Umsatzwachstum"]
        g_term = s.get(f"{seg.name} | TV-Wachstum", 0.0)
        g1 = year1_growth(g0, g_term, seg.fade_speed, seg.revenue_growth_mode == RevenueGrowthMode.FADE)
        m1 = s[f"{seg.name} | EBITDA-Marge"]                 # Jahr-1-Wert (Engine speichert arr[:, 0])
        out[seg.name] = {"wachstum": g1, "marge": m1}
        base.append(seg.base_revenue); gs.append(g1); ms.append(m1)
    g, m = aggregate(base, gs, ms)
    out["Konzern"] = {"wachstum": g, "marge": m}
    return out


def load_realisierungen(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(p, sep=";", dtype=str, keep_default_na=False)


def append_realisierung(path: str | Path, row: dict) -> None:
    p = Path(path)
    new = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def evaluate(real: pd.DataFrame, folder: str | Path) -> pd.DataFrame:
    """Je realisiertem Wert: Modell-P5/P50/P95 (in %) und Quantil-Rang (PIT)."""
    rows = []
    for _, z in real.iterrows():
        key = "Konzern" if z["ebene"] == "konzern" else z["segment"]
        try:
            dist = year1_distributions(str(Path(folder) / z["datei"]))[key][z["feld"]]
        except (KeyError, FileNotFoundError) as e:
            rows.append({**z.to_dict(), "fehler": f"nicht gefunden: {e}"})
            continue
        ist = float(z["ist_pct"]) / 100 if str(z["ist_pct"]).strip() not in ("", "nan") else float("nan")
        if not np.isfinite(ist):                                # fehlender Ist-Wert darf nie als Rang 0 zählen
            rows.append({**z.to_dict(), "fehler": "Ist-Wert fehlt"})
            continue
        rows.append({**z.to_dict(), "p5": 100 * np.percentile(dist, 5), "p50": 100 * np.median(dist),
                     "p95": 100 * np.percentile(dist, 95), "pit": pit(dist, ist), "fehler": ""})
    return pd.DataFrame(rows)


def convergence(register_csv: str | Path, prices_now: dict[str, float]) -> pd.DataFrame:
    """Je Firma und Stand: Lücke ln(P50/Kurs) damals und heute (heutiger Kurs) und geschlossener Anteil."""
    reg = pd.read_csv(register_csv, dtype=str, encoding="utf-8-sig")
    st = reg.drop_duplicates(["firma", "stand"])[["firma", "stand", "datum", "kurs", "p50"]].copy()
    st = st[st["firma"].isin(prices_now)]
    st["kurs"] = st["kurs"].astype(float); st["p50"] = st["p50"].astype(float)
    st = st[(st["kurs"] > 0) & (st["p50"] > 0)]
    st["kurs_heute"] = st["firma"].map(prices_now)
    st["luecke_damals"] = np.log(st["p50"] / st["kurs"])
    st["luecke_heute"] = np.log(st["p50"] / st["kurs_heute"])
    st["geschlossen"] = 1 - st["luecke_heute"] / st["luecke_damals"]
    return st.sort_values(["firma", "datum"])
