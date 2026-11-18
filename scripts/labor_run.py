"""Rückblick-Labor: alle Fälle bewerten (abbruchfest, setzt fort). Ergebnis: <SOTP_DATA>/labor_bewertungen.parquet.

Aufruf:  python scripts/labor_run.py [--jahre 2014-2020] [--n-sim 2000] [--limit N]
Wichtig: ungefiltertes Panel (load_panel filtert Firmen mit < 6 Jahren ⇒ Abgänge fielen raus = Überlebensverzerrung).
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.labor import basisraten, bewerte, faelle  # noqa: E402
from infrastructure.sec_xbrl import UA, build_lab_panel, data_dir, fetch_sic  # noqa: E402

DGS3 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3&cosd=2013-01-01"


def rf_am_stichtag() -> dict[int, float]:
    req = urllib.request.Request(DGS3, headers={"User-Agent": UA})
    s = pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=60).read().decode()), na_values=".")
    s = s.set_index(pd.to_datetime(s.iloc[:, 0])).iloc[:, 1].dropna() / 100
    return {y: float(s[:f"{y}-06-30"].iloc[-1]) for y in range(2013, 2026)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jahre", default="2014-2020")
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.jahre.split("-"))
    panel = pd.read_parquet(data_dir() / "panel.parquet")
    # SIC nur fürs Universum abrufen (Rest aus dem Cache, sonst „Sonstige“ in den Basisraten)
    panel = panel.assign(sic=panel.cik.map(fetch_sic(panel[panel.umsatz >= 5e8].cik.unique())))
    lab = build_lab_panel()
    rf = rf_am_stichtag()
    out = data_dir() / "labor_bewertungen.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    have = set(zip(done.cik, done.t)) if len(done) else set()
    rows = []
    for t in range(y0, y1 + 1):
        f, ob = faelle(panel, lab, t), basisraten(panel, t)
        f = f.head(a.limit) if a.limit else f
        print(f"{t}: {len(f)} Fälle, Basisraten n={len(ob)}, rf {rf[t]:.4f}", flush=True)
        t0 = time.time()
        for i, fall in f.iterrows():
            if (int(fall.cik), t) in have:
                continue
            try:
                rows.append(bewerte(fall, ob, t, rf[t], a.n_sim))
            except Exception as e:          # ein kaputter Fall stoppt nicht den Lauf
                rows.append({"cik": int(fall.cik), "t": t, "name": fall["name"], "fehler": str(e)[:200]})
            if len(rows) % 100 == 0:
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_parquet(out)
                rows = []
                print(f"  {i + 1}/{len(f)} ({time.time() - t0:.0f} s)", flush=True)
    if rows:
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out)
    print("fertig", out)


if __name__ == "__main__":
    main()
