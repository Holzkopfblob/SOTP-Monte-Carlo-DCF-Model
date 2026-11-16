"""Service-Test: Einfrieren bei unterdrückter Order und Neuverteilung (rebalance)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

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


def test_factor_scenarios_keep_rows_aligned(monkeypatch):
    """Bewertungen aus simulate_universe (Schlüssel „schock“): Zeilen sind gemeinsame Welten ⇒ kein Umsortieren,
    keine Markt-Welt, kein Repro-Hinweis (Neulauf mit Schocks weicht vom gespeicherten P50 ab)."""
    import application.portfolio_service as ps
    positions, valuations, closes = _setup()
    for v in valuations:
        v.update(schock={}, results=v["samples"], p50_gespeichert=v["p50_neu"] * 0.5)

    def boom(*a, **k):
        raise AssertionError("Altpfad im Faktor-Modus")
    monkeypatch.setattr(ps, "correlate_samples", boom)
    monkeypatch.setattr(ps, "mix_worlds", boom)
    _fake_irr(monkeypatch, {})
    res = rebalance(positions, valuations, closes, cash_eur=1100.0, p=Params(p_market=0.3), today=date(2026, 11, 14))
    assert not any("Neulauf" in w for w in res.warnings)
    assert res.table.set_index("Firma").loc["B", "IRR P50"] == 0.12


def _fake_irr(monkeypatch, rand: dict):
    """hold_returns ohne Engine: Rendite aus den Wert-Samples, IRR 12 %, *rand* = {Kurs: Anteil am Rand}."""
    import application.portfolio_service as ps

    def fake(results, price, k_d, h):
        return {"R": results / price - 1, "irr": 0.12, "irr_ek": 0.12, "d_e": 0.0, "irr_p5": 0.1, "irr_p95": 0.14,
                "am_rand": rand.get(round(price), 0.0)}
    monkeypatch.setattr(ps, "hold_returns", fake)


def test_irr_at_edge_is_frozen_and_sales_use_the_loss_pot(monkeypatch):
    """Faktor-Modus: Titel mit IRR am Rand (Bridge > Börsenwert) bleibt auf Ist-Gewicht; Verkaufsgewinne gegen den
    Aktienverlusttopf, Warnung erst, wenn sie ihn übersteigen."""
    positions, valuations, closes = _setup()
    positions[1]["stueck"], positions[0]["einstand_eur"] = 20.0, 60.0     # B gehalten; A mit Gewinn verkaufen
    for v in valuations:
        v.update(schock={}, results=v["samples"])
    _fake_irr(monkeypatch, {50: 0.5})                                     # B (Kurs 50) am Rand
    p = Params(min_order_eur=10.0)
    res = rebalance(positions, valuations, closes, cash_eur=0.0, p=p, today=date(2026, 11, 14), verlusttopf_eur=100.0)
    t = res.table.set_index("Firma")
    assert t.loc["B", "Ziel-Gewicht"] == t.loc["B", "Ist-Gewicht"] and t.loc["B", "Order Stück"] == 0
    assert any("Bridge" in w and "B" in w for w in res.warnings)
    assert t.loc["A", "Order Stück"] < 0                                  # A überbewertet ⇒ Verkauf
    gain = (100.0 - 60.0) * -t.loc["A", "Order Stück"]
    assert t.loc["A", "Gewinn Verkauf €"] == gain and res.steuer["topf_nachher"] == 100.0 - gain
    assert any("übersteigen den Verlusttopf" in w for w in res.warnings) == (gain > 100.0)


def test_usd_hold_returns_are_converted_by_interest_parity(monkeypatch):
    import application.portfolio_service as ps
    positions, valuations, closes = _setup()
    valuations[1]["waehrung"] = "USD"
    for v in valuations:
        v.update(schock={}, results=v["samples"])
    seen = {}

    def fake(results, price, k_d, h):
        seen[round(price)] = k_d
        return {"R": results / price - 1, "irr": 0.1, "irr_ek": 0.1, "d_e": 0.0, "irr_p5": 0.1, "irr_p95": 0.1,
                "am_rand": 0.0}
    monkeypatch.setattr(ps, "hold_returns", fake)

    def er(rf_usd):
        p = Params(rf=0.03, rf_usd=rf_usd, horizon_years=10)
        res = rebalance(positions, valuations, closes, cash_eur=1000.0, p=p, today=date(2026, 11, 14))
        return res.table.set_index("Firma")["E[R] Horizont"]
    same, high = er(0.03), er(0.05)
    assert high["A"] == same["A"]                                        # EUR-Titel unberührt
    assert 1 + high["B"] == pytest.approx((1 + same["B"]) * (1.03 / 1.05) ** 10)
    assert seen[50] == pytest.approx((0.05 + 0.015) * 0.75) and seen[100] == pytest.approx((0.03 + 0.015) * 0.75)


def test_sector_bias_is_shared_within_a_sector(monkeypatch):
    """τ > 0: Titel derselben Branche bekommen in jeder Zeile denselben IRR-Fehler, andere Branchen einen eigenen."""
    positions, valuations, closes = _setup()
    for v, s in zip(valuations, ("Telko", "Telko", "Auto")):
        v.update(schock={"sektor": s}, results=np.full(len(v["samples"]), v["p50_neu"]))   # ohne Cashflow-Streuung
    _fake_irr(monkeypatch, {})
    import application.portfolio_service as ps
    seen = {}
    real = ps.kelly_weights

    def spy(R, *a, **k):
        seen["R"] = R
        return real(R, *a, **k)
    monkeypatch.setattr(ps, "kelly_weights", spy)
    rebalance(positions, valuations, closes, 1000.0, Params(tau_branche=0.03, horizon_years=10), date(2026, 11, 14))
    lr = np.log1p(seen["R"])
    assert np.corrcoef(lr[:, 0], lr[:, 1])[0, 1] > 0.99 and abs(np.corrcoef(lr[:, 0], lr[:, 2])[0, 1]) < 0.05
    assert np.std(lr[:, 0]) == pytest.approx(10 * 0.03 / 1.12, rel=0.05)     # τ über 10 J. verkettet


def test_rate_shock_hits_levered_titles_together_and_helps_net_cash(monkeypatch):
    import application.portfolio_service as ps
    positions, valuations, closes = _setup()
    for v in valuations:
        v.update(schock={}, results=np.full(len(v["samples"]), v["p50_neu"]))
    lev = {100: 1.0, 50: 2.0, 80: -0.5}                                  # A, B verschuldet; C Nettokasse

    def fake(results, price, k_d, h):
        return {"R": results / price - 1, "irr": 0.1, "irr_ek": 0.1, "d_e": lev[round(price)], "irr_p5": 0.1,
                "irr_p95": 0.1, "am_rand": 0.0}
    monkeypatch.setattr(ps, "hold_returns", fake)
    seen, real = {}, ps.kelly_weights
    monkeypatch.setattr(ps, "kelly_weights", lambda R, *a, **k: seen.setdefault("R", R) is R and real(R, *a, **k))
    rebalance(positions, valuations, closes, 1000.0, Params(tau_branche=0.0, sigma_zins=0.02, horizon_years=10),
              date(2026, 11, 14))
    lr = np.log1p(seen["R"])
    assert np.corrcoef(lr[:, 0], lr[:, 1])[0, 1] > 0.99 and np.corrcoef(lr[:, 0], lr[:, 2])[0, 1] < -0.99
    assert np.std(lr[:, 1]) == pytest.approx(2 * np.std(lr[:, 0]), rel=0.05)   # doppelter Hebel, doppelter Schaden
    from domain.factor_model import _rng
    zins = _rng(Params().seed, "zins").standard_normal(len(lr))
    assert np.corrcoef(lr[:, 0], zins)[0, 1] < -0.99                              # Zinsanstieg schadet Verschuldeten
