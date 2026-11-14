"""Service-Test: Einfrieren bei unterdrückter Order und Neuverteilung (rebalance)."""
from datetime import date

import numpy as np
import pandas as pd

from application.portfolio_service import Params, rebalance


def _setup():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-11-01", "2026-11-13")
    px = {"A": 100.0, "B": 50.0, "C": 80.0}
    closes = pd.DataFrame({t: p * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))) - 0)  # Random Walk
                           for t, p in px.items()}, index=idx)
    closes = closes / closes.iloc[-1] * pd.Series(px)          # letzter Kurs = px
    closes["EURUSD=X"] = 1.1
    n = 20_000

    def val(med):
        s = rng.lognormal(np.log(med), 0.15, n)
        return {"samples": s, "p50_neu": float(np.median(s)), "p50_gespeichert": None, "waehrung": "EUR",
                "stichtag": "2026-11-10", "kurs_recherche": None, "stand": ""}

    positions = [{"firma": "A", "ticker": "A", "stueck": 9.0},   # 900 € überbewertet, Verkauf < Mindestorder
                 {"firma": "B", "ticker": "B", "stueck": 0.0},   # attraktiv
                 {"firma": "C", "ticker": "C", "stueck": 0.0}]   # etwas weniger attraktiv
    valuations = [val(70.0), val(90.0), val(120.0)]
    return positions, valuations, closes


def test_suppressed_sell_is_frozen_and_cash_goes_to_best_buy():
    positions, valuations, closes = _setup()
    res = rebalance(positions, valuations, closes, cash_eur=1100.0,
                    p=Params(max_w=0.6, min_order_eur=1000.0), today=date(2026, 11, 14))
    t = res.table.set_index("Firma")
    assert t.loc["A", "Order Stück"] == 0 and t.loc["A", "Ziel-Gewicht"] == t.loc["A", "Ist-Gewicht"]
    assert 1100.0 + t.loc["Cash (EUR)", "Order €"] >= -1e-6            # kein Überziehen
    bought = t.loc[["B", "C"], "Order €"]
    assert (bought > 0).sum() >= 1 and bought.max() >= 1000.0          # Cash wird investiert, nicht liegen gelassen
    assert any("Mindestorder" in w for w in res.warnings)
