"""Portfolio-Seite: Klick auf „Zielgewichte berechnen“ erreicht rebalance mit allen Eingaben (Regressionstest für den
NameError etf_stueck vom 18.11.2026). Schwere Teile (Simulation, Kurse, Zinsen, Optimierer) sind Attrappen, kein Netz."""
import json

from streamlit.testing.v1 import AppTest


def _page():
    import streamlit as st

    import presentation.pages.portfolio as pf

    def stub_rebalance(positions, valuations, closes, cash, p, **kw):
        raise ValueError(f"stub erreicht: cash={cash} topf={kw.get('verlusttopf_eur')} etf={kw.get('etf_stueck')} "
                         f"ks={[q.get('kaufsteuer_pct') for q in positions]}")
    pf._bund_3y = lambda: (0.03, "test")
    pf._treasury_3y = lambda: (0.045, "test")
    pf._erp = lambda: (0.0441, "2026-11-01", True)
    pf._simulate_universe = lambda items, mtimes: [{"stichtag": "2026-11-01"} for _ in items]
    pf._simulate = lambda path, mtime: {"stichtag": "2026-11-01"}
    pf._closes = lambda tickers: None
    pf.rebalance = stub_rebalance
    pf.render_portfolio(st.container())


def _berechnen(tmp_path, monkeypatch, **extra) -> list[str]:
    bew = tmp_path / "bewertungen"
    bew.mkdir()
    (bew / "A_2026-11-01.json").write_text(json.dumps(
        {"metadata": {"company": "A AG", "stichtag": "2026-11-01"}, "ergebnis": {"preis_je_aktie": {"p50": 1.0}}}))
    depot = tmp_path / "depot.json"
    depot.write_text(json.dumps({
        "bewertungen_ordner": str(bew), "cash_eur": 1000.0, "verlusttopf_eur": 50.0, "etf_stueck": 2.0,
        "positionen": [{"firma": "A AG", "ticker": "A", "stueck": 1.0, "datei": "", "einstand_eur": None,
                        "kaufdatum": "", "gruppe": "", "sektor": "", **extra}], "transaktionen": []}))
    monkeypatch.setenv("SOTP_DEPOT", str(depot))
    at = AppTest.from_function(_page, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if "Zielgewichte" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    return [e.value for e in at.error]


def test_berechnen_reicht_alle_eingaben_an_rebalance(tmp_path, monkeypatch):
    err = _berechnen(tmp_path, monkeypatch)
    assert any("stub erreicht: cash=1000.0 topf=50.0 etf=2.0 ks=[None]" in e for e in err), err


def test_kaufsteuer_kommt_bei_rebalance_an(tmp_path, monkeypatch):
    """Regression 19.11.2026: die Seite baute Positionen aus fester Spaltenliste, kaufsteuer_pct fiel weg."""
    err = _berechnen(tmp_path, monkeypatch, kaufsteuer_pct=0.4)
    assert any("ks=[0.4]" in e for e in err), err
