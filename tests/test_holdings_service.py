"""Bestandssicht: Gewinn, Aktion, Kandidaten, Termine und These aus der Research-Ablage."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from application.holdings_service import holdings, next_events, thesis


def _table():
    return pd.DataFrame({
        "Firma": ["A", "B", "C", "Cash (EUR)"], "Kurs €": [12.0, 50.0, 20.0, np.nan],
        "Ist-Gewicht": [0.4, 0.3, 0.0, 0.3], "Ziel-Gewicht": [0.0, 0.3, 0.1, 0.6],
        "Wert P50 €": [10.0, 80.0, 30.0, np.nan], "Upside P50": [-0.17, 0.6, 0.5, np.nan],
        "Rendite p. a.": [0.05, 0.12, 0.10, np.nan], "Beta": [1.0, 0.5, 0.8, np.nan],
        "Vorsprung p. a.": [-0.02, 0.06, 0.03, np.nan], "Order €": [-120.0, 0.0, 0.0, np.nan],
        "Umsetzung": ["Order", "", "Sparplan", np.nan], "Sparplan €/Monat": [np.nan, np.nan, 5.0, np.nan],
        "Stand": ["s1", "s2", "s3", np.nan]})


def test_bestand_gewinn_aktion_und_kandidaten(tmp_path):
    (tmp_path / "Firmen").mkdir()
    (tmp_path / "termine.csv").write_text("ticker;datum;ereignis;aktion;status\nB;2026-12-01;Q4;x;offen\n"
                                          "B;2026-11-20;CMD;y;fest\nA;2026-10-01;alt;z;offen\nA;2026-09-01;e;z;erledigt\n",
                                          encoding="utf-8")
    (tmp_path / "Firmen" / "B.md").write_text("# B\n\n## These\n_Entwurf, bitte prüfen._\n\nB ist billig. Weil X.\n\n"
                                              "## Auslöser\n", encoding="utf-8")
    pos = [{"firma": "A", "ticker": "A", "stueck": 10, "einstand_eur": 10.0, "kaufdatum": "2026-11-08"},
           {"firma": "B", "ticker": "B", "stueck": 2, "einstand_eur": None, "kaufdatum": ""},
           {"firma": "C", "ticker": "C", "stueck": 0, "einstand_eur": None, "kaufdatum": ""}]
    held, new, tot = holdings(_table(), pos, 100.0, today=date(2026, 11, 18), research=tmp_path)
    a = held.set_index("Firma")
    assert a.loc["A", "Gewinn €"] == pytest.approx(20.0) and a.loc["A", "Gewinn %"] == pytest.approx(0.2)
    assert a.loc["A", "Aktion"] == "Alles verkaufen" and a.loc["B", "Aktion"] == "Halten"
    assert a.loc["A", "Haltedauer (Tage)"] == 10
    assert a.loc["B", "Nächster Termin"] == "20.11. CMD" and a.loc["A", "Nächster Termin"].startswith("⚠")
    assert a.loc["B", "These"] == "B ist billig."
    assert list(new["Firma"]) == ["C"] and new.loc[0, "Aktion"] == "Sparplan 5 €/Monat"
    assert tot["wert"] == pytest.approx(220.0) and tot["gesamt"] == pytest.approx(320.0)
    assert tot["einstand"] == pytest.approx(100.0) and tot["ohne_einstand"] == ["B"]


def test_ohne_ablage_leer(tmp_path):
    assert next_events(tmp_path) == {} and thesis("X", tmp_path) == ""


def test_bestand_mit_transaktionsbuch_und_geschlossener_position(tmp_path):
    from application.holdings_service import closed_positions
    tx = [{"datum": "2026-11-01", "ticker": "A", "typ": "kauf", "stueck": 10, "kurs_eur": 10.0, "gebuehr_eur": 1.0},
          {"datum": "2026-11-10", "ticker": "A", "typ": "dividende", "betrag_eur": 3.0},
          {"datum": "2026-10-01", "ticker": "Z", "typ": "kauf", "stueck": 2, "kurs_eur": 50.0, "gebuehr_eur": 1.0},
          {"datum": "2026-11-05", "ticker": "Z", "typ": "verkauf", "stueck": 2, "kurs_eur": 60.0, "gebuehr_eur": 1.0}]
    pos = [{"firma": "A", "ticker": "A", "stueck": 10, "einstand_eur": 10.1, "kaufdatum": "2026-11-01"},
           {"firma": "Zeta", "ticker": "Z", "stueck": 0, "einstand_eur": None, "kaufdatum": ""}]
    t = _table().iloc[[0, 3]].reset_index(drop=True)
    held, _, tot = holdings(t, pos, 0.0, today=date(2026, 11, 18), research=tmp_path, transaktionen=tx)
    a = held.set_index("Firma").loc["A"]
    assert a["Dividenden €"] == 3.0 and a["Gebühren €"] == 1.0
    assert a["Ergebnis gesamt €"] == pytest.approx((12.0 - 10.1) * 10 + 3.0)
    assert tot["realisiert"] == pytest.approx(2 * 60 - 1 - 101) and tot["dividenden"] == 3.0
    c = closed_positions(pos, tx)
    assert list(c["Firma"]) == ["Zeta"] and c.loc[0, "Ergebnis €"] == pytest.approx(18.0)
