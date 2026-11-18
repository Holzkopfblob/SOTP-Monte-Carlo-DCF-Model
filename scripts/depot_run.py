"""Depot-Optimierung ohne UI (wie die Portfolio-Seite im Faktor-Modus, 10 J. auf IRR-Basis).

Aufruf (aus dem Repo, mit .venv):
  python scripts/depot_run.py <depot.json> [--kappa 0,0.005,0.01] [--detail 0.01] [--n 20000] [--min-order 100]
Mehrere κ ⇒ Vergleichstabelle der Ziel-Gewichte; --detail κ ⇒ Orders und Sparpläne für dieses κ.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from application.portfolio_service import Params, find_valuations, load_depot, rebalance  # noqa: E402
from application.universe_simulation import simulate_universe  # noqa: E402
from infrastructure.market_data import fetch_bund_3y, fetch_closes, fetch_treasury_3y  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("depot")
    ap.add_argument("--kappa", default="0,0.005,0.01,0.02,0.05")
    ap.add_argument("--detail", type=float)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--min-order", type=float, default=Params.min_order_eur)
    a = ap.parse_args()

    depot = load_depot(a.depot)
    pos = depot["positionen"]
    vals = find_valuations(depot["bewertungen_ordner"])
    folder = Path(depot["bewertungen_ordner"])
    files = [folder / p["datei"] if p.get("datei") else Path(vals[p["firma"]][-1]) for p in pos]
    rf, rf_usd = fetch_bund_3y()[0], fetch_treasury_3y()[0]
    closes = fetch_closes(sorted({p["ticker"] for p in pos}))
    v = simulate_universe([(p["firma"], f, p.get("sektor")) for p, f in zip(pos, files)], n=a.n, seed=0)
    print(f"rf EUR {rf:.4f}, USD {rf_usd:.4f}; {len(pos)} Titel; Bewertungen: "
          + ", ".join(f"{p['ticker']}={f.stem.split('_')[1]}" for p, f in zip(pos, files)))

    def run(k):
        p = Params(kappa=k, horizon_years=10, rf=rf, rf_usd=rf_usd, faktor_szenarien=True, min_order_eur=a.min_order)
        return rebalance(pos, v, closes, depot["cash_eur"], p, date.today(), depot.get("verlusttopf_eur", 0.0))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    ks = [float(x) for x in a.kappa.split(",") if x]
    cols, foot = {}, {}
    for k in ks:
        r = run(k)
        t = r.table.set_index("Firma")
        cols[f"κ={k:g}"] = t["Ziel-Gewicht"]
        foot[f"κ={k:g}"] = {"Titel > 0,5 %": int((t["Ziel-Gewicht"].iloc[:-1] > 0.005).sum()),
                            "10 J. P5 €": r.wealth_ziel["p5"], "10 J. P50 €": r.wealth_ziel["p50"],
                            "Sparpläne": int((t["Umsetzung"] == "Sparplan").sum())}
        base = t
    tab = pd.DataFrame(cols)
    tab.insert(0, "Vorsprung p. a.", base["Vorsprung p. a."])
    tab.insert(0, "Ist", base["Ist-Gewicht"])
    print("\n=== Ziel-Gewichte je κ ===")
    print(tab.map(lambda x: f"{x:6.1%}" if pd.notna(x) else "").to_string())
    print(pd.DataFrame(foot).round(0).to_string())

    if a.detail is not None:
        r = run(a.detail)
        t = r.table
        print(f"\n=== Detail κ = {a.detail:g} ===")
        print(t[["Firma", "IRR EK", "Vorsprung p. a.", "Ist-Gewicht", "Ziel-Gewicht", "Order €", "Umsetzung",
                 "Sparplan €/Monat"]].round(4).to_string(index=False))
        for w in r.warnings:
            print("  ⚠", w)


if __name__ == "__main__":
    main()
