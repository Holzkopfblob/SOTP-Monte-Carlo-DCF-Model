import numpy as np
import pandas as pd
import pytest

from domain.base_rates import conditional, outcomes, percentile_of, sektor


def _panel():
    rows = []
    for cik in range(300):                       # 300 Telkos mit 2 %/Jahr, Marge 40 % → 42 %
        for j, year in enumerate(range(2010, 2021)):
            rows.append(dict(cik=cik, jahr=year, umsatz=1e9 * 1.02 ** j, ebit=0.25e9 * 1.02 ** j,
                             da=(0.15 + 0.002 * j) * 1e9 * 1.02 ** j, sic="4813"))
    for cik in range(300, 330):                  # 30 Software-Firmen mit 20 %/Jahr (zu wenige ⇒ Lockerung)
        for j, year in enumerate(range(2010, 2021)):
            rows.append(dict(cik=cik, jahr=year, umsatz=2e8 * 1.2 ** j, ebit=0.0, da=0.0, sic="7372"))
    return pd.DataFrame(rows)


def test_cagr_and_margin_change_over_horizon():
    o = outcomes(_panel(), 5)
    tel = o[o.sektor == "Telko"]
    assert np.allclose(tel.cagr, 0.02) and np.allclose(tel.dm, 0.01)       # 5 J. × 0,2 pp Marge
    assert set(tel.g_start) == {-1, 1}                                     # erstes Jahr ohne Vorjahr; sonst Klasse 1 = 0–5 %
    assert len(tel[tel.cik == 0]) == 6                                     # Startjahre 2010–2015 haben 2015–2020


def test_conditional_loosens_when_too_few_cases():
    o = outcomes(_panel(), 3)
    vals, used = conditional(o, "cagr", min_n=200, sektor="Telko", g_start=1)
    assert used == {"sektor": "Telko", "g_start": 1} and np.allclose(vals, 0.02)
    # Keine Telko mit 20–40 % Vorjahreswachstum ⇒ nur g_start lockern, Branche behalten
    vals, used = conditional(o, "cagr", min_n=200, sektor="Telko", g_start=4)
    assert used == {"sektor": "Telko"} and np.allclose(vals, 0.02)
    # Software: 30 Firmen × 8 Startjahre = 240 < 300 ⇒ auch die Branche fällt
    vals, used = conditional(o, "cagr", min_n=300, sektor="Software/IT")
    assert used == {} and len(vals) == len(o)
    assert percentile_of(np.array([0.0, 0.1, 0.2, 0.3]), 0.15) == pytest.approx(0.5)


def test_sector_mapping():
    assert sektor("4813") == "Telko" and sektor(3674) == "Halbleiter/Elektronik" and sektor(None) == "Sonstige"
    assert sektor("6021") == "Finanz" and sektor("7372") == "Software/IT"
