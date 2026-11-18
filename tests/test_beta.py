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


def test_ols_2f_findet_beide_steigungen():
    from domain.beta import ols_2f
    rng = np.random.default_rng(3)
    m, z = rng.normal(0, 0.02, 400), rng.normal(0, 0.005, 400)
    o = ols_2f(0.8 * m + 3.0 * z + rng.normal(0, 0.01, 400), m, z)
    assert abs(o["beta"] - 0.8) < 3 * o["se"] and abs(o["beta_z"] - 3.0) < 3 * o["se_z"]


def test_shrink_to_zero_drueckt_rauschen_laesst_echten_effekt():
    from domain.beta import shrink_to_zero
    b = np.array([0.1, -0.2, 0.15, -0.05, 4.6])      # vier Rauschen, ein echter Effekt (Vonovia-artig)
    se = np.array([0.5, 0.5, 0.5, 0.5, 0.48])
    post, w = shrink_to_zero(b, se)
    assert np.all(np.abs(post[:4]) < 0.15) and post[4] > 3.0 and 0 < w[0] < 1
    assert np.all(shrink_to_zero(np.array([0.1, -0.1, 0.05]), np.array([1.0, 1.0, 1.0]))[0] == 0)  # nur Rauschen ⇒ 0


def test_equity_beta_ist_relevered_branchenbeta():
    from domain.beta import debt_beta, equity_beta, industry_asset_beta, relever
    b, _ = equity_beta(None, "EUR", 0.74, "Real Estate (Operations & Services)")
    b_a, _ = industry_asset_beta(None, "EUR", "Real Estate (Operations & Services)")
    assert b == pytest.approx(relever(b_a, 0.74, debt_beta(0.74))) and b > b_a


def test_treynor_black_mit_index_gibt_rest_an_den_etf():
    from domain.beta import treynor_black_anchor
    a = treynor_black_anchor(np.array([0.02, 0.0, -0.01]), np.array([0.04, 0.04, 0.04]), budget=0.98,
                             passiv=(0.04, 0.02))
    assert a[0] == pytest.approx((0.02 / 0.04) / (0.04 / 0.02))       # 0,25
    assert a[1] == 0 and a[2] == 0 and a[3] == pytest.approx(0.98 - 0.25) and a[4] == pytest.approx(0.02)
    assert a.sum() == pytest.approx(1.0)
