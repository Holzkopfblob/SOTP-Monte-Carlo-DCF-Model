"""Rückblick-Labor: Fallauswahl und Aktionärsrendite auf Spielzeugdaten (ohne SEC-Abruf)."""
import numpy as np
import pandas as pd
import pytest

from application.labor import ergebnisse, faelle


def _daten():
    rows, lab = [], []
    for cik, sic in ((1, "3571"), (2, "6020"), (3, "3571")):          # 2 = Bank ⇒ raus
        for y in range(2010, 2021):
            rows.append({"cik": cik, "jahr": y, "name": f"F{cik}", "sic": sic, "umsatz": 1e9 * 1.05 ** (y - 2010),
                         "ebit": 1e8, "da": 5e7, "capex": 6e7})
            lab.append({"cik": cik, "jahr": y, "schulden": 3e8, "kasse": 1e8, "nettoschulden": 2e8, "dividenden": 1e7, "rueckkauf": 0.0,
                        "emission": 0.0, "public_float": (2e9 if cik != 3 or y <= 2016 else np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(lab)


def test_faelle_ohne_finanzwerte_mit_float():
    panel, lab = _daten()
    f = faelle(panel, lab, 2015)
    assert sorted(f.cik) == [1, 3] and f.marge.iloc[0] == pytest.approx(1.5e8 / (1e9 * 1.05 ** 4))
    assert f.g_vj.iloc[0] == pytest.approx(0.05)


def test_ergebnisse_ausschuettung_und_abgang():
    panel, lab = _daten()
    o = ergebnisse(panel, lab, 2015, h=5).set_index("cik")
    # Firma 1: Float konstant 2 Mrd., 5 × 10 Mio. Dividende ⇒ TSR = 50 Mio. / 2 Mrd. = 2,5 %
    assert o.loc[1, "tsr_roh"] == pytest.approx(0.025) and not o.loc[1, "abgang"]
    assert o.loc[1, "umsatz_cagr5"] == pytest.approx(0.05)
    # Firma 3: letzter Float 2016 ⇒ Abgang, nur Ausschüttung 2015 zählt
    assert o.loc[3, "abgang"] and o.loc[3, "ende_jahr"] == 2016 and o.loc[3, "tsr_roh"] == pytest.approx(0.005)
