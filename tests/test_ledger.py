"""Transaktionsbuch: FIFO wie die Bank, Gebühren in Anschaffungs- und Veräußerungskosten, Dividenden, Teilverkäufe."""
import pytest

from domain.ledger import apply_to_positions, book


def T(datum, typ, stueck=0, kurs=0, gebuehr=0, betrag=0, ticker="X", steuer=0):
    return {"datum": datum, "ticker": ticker, "typ": typ, "stueck": stueck, "kurs_eur": kurs,
            "gebuehr_eur": gebuehr, "betrag_eur": betrag, "steuer_eur": steuer}


def test_fifo_teilverkauf_mit_gebuehren():
    k = book([T("2026-01-02", "kauf", 10, 100, 1), T("2026-03-01", "kauf", 10, 120, 1),
              T("2026-06-01", "verkauf", 15, 130, 1)])["X"]
    # FIFO: 10 × 10,10 + 5 × 12,01... Kosten je Stück Lot 1 = 100,1, Lot 2 = 120,1
    assert k.realisiert == pytest.approx(15 * 130 - 1 - (10 * 100.1 + 5 * 120.1))
    assert k.stueck == pytest.approx(5) and k.einstand == pytest.approx(120.1)
    assert k.erster_kauf_offen == "2026-03-01" and k.gebuehren == pytest.approx(3)


def test_geschlossene_position_dividende_und_sonstige_gebuehr():
    k = book([T("2026-10-31", "kauf", 4, 250, 1), T("2026-11-10", "dividende", betrag=2.5),
              T("2026-11-16", "verkauf", 4, 266, 1), T("2026-11-17", "gebuehr", betrag=0.5)])["X"]
    assert k.geschlossen and k.stueck == pytest.approx(0)
    assert k.realisiert == pytest.approx(4 * 266 - 1 - 1001) and k.dividenden == 2.5
    assert k.gebuehren == pytest.approx(2.5)


def test_reihenfolge_nach_datum_und_ueberverkauf_warnt():
    k = book([T("2026-02-01", "verkauf", 5, 10), T("2026-01-01", "kauf", 3, 8)])["X"]
    assert k.stueck == 0 and any("mehr als im Bestand" in w for w in k.warnungen)


def test_apply_ueberschreibt_nur_gebuchte_ticker():
    pos = [{"firma": "A", "ticker": "X", "stueck": 99, "einstand_eur": 1.0, "kaufdatum": ""},
           {"firma": "B", "ticker": "Y", "stueck": 2, "einstand_eur": 5.0, "kaufdatum": "2026-01-01"}]
    out, _ = apply_to_positions(pos, [T("2026-01-02", "kauf", 3, 10, 1)])
    assert out[0]["stueck"] == 3 and out[0]["einstand_eur"] == pytest.approx(31 / 3, abs=1e-4)
    assert out[1] == pos[1]
