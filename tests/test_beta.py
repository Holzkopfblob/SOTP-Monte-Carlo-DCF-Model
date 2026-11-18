"""Beta: OLS, Vasicek-Schrumpfung, Harris–Pringle, Treynor–Black-Anker."""
import numpy as np
import pytest

from domain.beta import debt_beta, industry_asset_beta, ols_beta, relever, treynor_black_anchor, unlever, vasicek


def test_ols_trifft_wahres_beta_und_restrisiko():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.02, 5000)
    eps = rng.normal(0, 0.03, 5000)
    o = ols_beta(0.001 + 0.7 * m + eps, m)
    assert o["beta"] == pytest.approx(0.7, abs=0.03)
    assert o["resid_var"] == pytest.approx(0.03 ** 2 * 52, rel=0.05)


def test_vasicek_zieht_unsichere_schaetzung_zum_prior():
    b_sicher, w_sicher = vasicek(1.4, 0.05, 1.0)
    b_unsicher, w_unsicher = vasicek(1.4, 0.6, 1.0)
    assert w_sicher > 0.9 > 0.3 > w_unsicher
    assert 1.0 < b_unsicher < b_sicher < 1.4
    assert vasicek(np.nan, np.inf, 0.8) == (0.8, 0.0)


def test_harris_pringle_hin_und_zurueck():
    for d_v in (0.0, 0.3, 0.6):
        bd = debt_beta(d_v)
        assert relever(unlever(0.9, d_v, bd), d_v, bd) == pytest.approx(0.9)
    assert relever(0.4, 0.5, 0.1) == pytest.approx(0.7)          # Hebel erhöht das Equity-Beta


def test_damodaran_prior_telko_europa():
    b_a, src = industry_asset_beta("Telko", "EUR")
    assert 0.35 < b_a < 0.5 and "Europa Telecom. Services" in src
    assert industry_asset_beta("unbekannt", "USD")[1].startswith("Damodaran US Total Market")


def test_treynor_black_anker():
    a = treynor_black_anchor(np.array([0.04, 0.02, -0.01]), np.array([0.04, 0.01, 0.02]), budget=1.0)
    assert a.sum() == pytest.approx(1.0) and a[2] == 0.0
    assert a[1] == pytest.approx(2 * a[0])          # halbes Alpha, aber ein Viertel Restrisiko ⇒ doppelter Anker
