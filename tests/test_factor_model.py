import numpy as np
import pandas as pd
import pytest

from domain.factor_model import draw_shocks, fit_phi, firm_params, growth_ratio, margin_ratio, robust_sd
from domain.models import DistributionConfig as D, SegmentConfig, SimulationConfig, TerminalValueMethod
from domain.valuation import compute_segment_ev
from infrastructure.monte_carlo_engine import MonteCarloEngine

COMMON = {"g": {"makro": 0.03, "Telko": 0.04}, "m": {"makro": 0.0, "Telko": 0.0}}


def _firm(name, sektor, sigma=0.10, phi=0.4):
    p = {"sigma": sigma, "phi": phi, "beta": ((0.03 / sigma) ** 2, (0.04 / sigma) ** 2 if sektor == "Telko" else 0.0)}
    return {"name": name, "sektor": sektor, "g": p, "m": {"sigma": 0.02, "phi": 0.9, "beta": (0.0, 0.0)}}


def test_ratios_and_phi_fit_roundtrip():
    assert growth_ratio(0.0, 5) == 5 and growth_ratio(1.0, 5) == 55 and margin_ratio(1.0, 4) == 4
    for phi in (0.0, 0.43, 0.96):
        assert fit_phi(growth_ratio, growth_ratio(phi, 5), 5) == pytest.approx(phi, abs=1e-6)
        assert fit_phi(margin_ratio, margin_ratio(phi, 4), 4) == pytest.approx(phi, abs=1e-6)
    assert fit_phi(growth_ratio, 1.0, 5) == 0.0 and fit_phi(margin_ratio, 9.0, 4) == 1.0


def test_shocks_reproduce_the_moments_they_were_fitted_to():
    """Simulierte Basisrate (AR(1), φ = 0,43) → firm_params → draw_shocks trifft Streuung über 1 und 5 Jahre."""
    rng = np.random.default_rng(3)
    eta = rng.standard_normal((200_000, 5)) * 0.09
    s = np.zeros_like(eta)
    for t in range(5):
        s[:, t] = (0.43 * s[:, t - 1] if t else 0) + eta[:, t]
    f = firm_params("X", "Telko", s[:, 0], s.sum(1), 5, eta[:, 0] / 4, eta[:, :4].sum(1) / 4, 4, COMMON)
    assert f["g"]["sigma"] == pytest.approx(0.09, rel=0.02) and f["g"]["phi"] == pytest.approx(0.43, abs=0.03)
    assert f["m"]["phi"] == pytest.approx(1.0, abs=0.03)                     # Random Walk der Marge
    g = draw_shocks([f], 100_000, 6, seed=1)[0]["g"]
    assert robust_sd(g[:, :5].sum(1)) == pytest.approx(robust_sd(s.sum(1)), rel=0.03)


def test_same_sector_firms_share_factor_paths():
    """Korrelation der 5-J.-Umsatzschocks = Anteil gemeinsamer Innovationen (gleiches φ ⇒ exakt β_makro + β_branche)."""
    sh = draw_shocks([_firm("A", "Telko"), _firm("B", "Telko"), _firm("C", "Software/IT")], 100_000, 5, seed=7)
    lvl = [x["g"].sum(1) for x in sh]
    assert np.corrcoef(lvl[0], lvl[1])[0, 1] == pytest.approx(0.09 + 0.16, abs=0.02)
    assert np.corrcoef(lvl[0], lvl[2])[0, 1] == pytest.approx(0.09, abs=0.02)
    # Reihenfolge der Titel ändert die Welt nicht (eigene Ströme je Faktor/Firma)
    again = draw_shocks([_firm("C", "Software/IT"), _firm("A", "Telko")], 100_000, 5, seed=7)
    assert np.array_equal(again[1]["g"], sh[0]["g"])


def _ev(**kw):
    n, T = 4, 5
    one = np.ones(n)
    return compute_segment_ev(1000.0, T, 0.03 * one, 0.30 * one, 0.05 * one, 0.25 * one, 0.05 * one, 0.0 * one,
                              0.08 * one, kw.pop("method", TerminalValueMethod.EXIT_MULTIPLE), 0.02 * one, 10.0 * one,
                              mid_year_convention=False, decompose=True, **kw)


def test_growth_shock_scales_revenue_level_and_margin_shock_decays_in_tv():
    base = _ev()
    up = _ev(g_shock=np.full((4, 5), 0.0) + np.array([np.log(1.1), 0, 0, 0, 0]))
    assert np.allclose(up.revenue, base.revenue * 1.1) and np.allclose(up.tv_abs, base.tv_abs * 1.1)
    m = np.full((4, 5), 0.02)
    full = _ev(m_shock=m, m_phi=1.0)                     # dauerhaft ⇒ Exit-EBITDA um 2 Punkte höher
    assert np.allclose(full.tv_abs - base.tv_abs, 10 * 0.02 * base.revenue[:, -1])
    none = _ev(m_shock=m, m_phi=0.0)                     # vorübergehend ⇒ TV unverändert, Jahre 1–5 höher
    assert np.allclose(none.tv_abs, base.tv_abs) and (none.fcff > base.fcff).all()
    g = _ev(method=TerminalValueMethod.GORDON_GROWTH, m_shock=m, m_phi=1.0)
    g0 = _ev(method=TerminalValueMethod.GORDON_GROWTH)
    assert np.allclose(g.tv_num - g0.tv_num, 0.02 * 0.75 * g0.revenue[:, -1] * 1.02)


def test_engine_applies_firm_shocks_to_every_segment():
    segs = [SegmentConfig(name=k, base_revenue=500.0, forecast_years=T, revenue_growth=D(fixed_value=0.03),
                          ebitda_margin=D(fixed_value=0.3)) for k, T in (("A", 5), ("B", 7))]
    cfg = SimulationConfig(n_simulations=500, random_seed=1, segments=segs)
    base = MonteCarloEngine(cfg).run()
    sh = {"g": np.full((500, 7), 0.01), "m": np.full((500, 7), 0.05), "m_phi": 1.0}
    res = MonteCarloEngine(cfg, sh).run()
    for k, T in (("A", 5), ("B", 7)):
        r, r0 = res.segment_cashflows[k], base.segment_cashflows[k]
        assert np.allclose(r.revenue[:, -1] / r0.revenue[:, -1], np.exp(0.01 * T))
        assert np.allclose(r.ebitda / r.revenue, 0.35)
    with pytest.raises(ValueError):
        MonteCarloEngine(cfg, {"g": np.zeros((500, 5)), "m": np.zeros((500, 5)), "m_phi": 1.0}).run()


def test_shock_params_take_both_moments_from_the_same_cohort():
    """5-J.-Horizont hat zu wenige große Firmen ⇒ Größe fällt; der 1-J.-Horizont muss dieselbe Kohorte nutzen,
    obwohl er für große Firmen allein genug Fälle hätte (sonst σ_1 eng, σ_5 breit ⇒ φ zu hoch)."""
    from application.universe_simulation import shock_params
    rng = np.random.default_rng(0)
    def o(n_gross, n_klein):
        return pd.DataFrame({"sektor": "Telko", "groesse": [2] * n_gross + [0] * n_klein, "g_start": 1, "m_start": 2,
                             "cagr": np.r_[rng.normal(0.02, 0.03, n_gross), rng.normal(0.02, 0.15, n_klein)],
                             "dm": np.r_[rng.normal(0, 0.01, n_gross), rng.normal(0, 0.05, n_klein)]})
    ob = {1: o(500, 500), 5: o(50, 500), 4: o(50, 500)}
    cfg = SimulationConfig(n_simulations=10, segments=[SegmentConfig(name="A", base_revenue=20_000.0,
                           revenue_growth=D(fixed_value=0.02), ebitda_margin=D(fixed_value=0.35))])
    f = shock_params("X", cfg, "Telko", "USD", ob=ob, common=COMMON)
    assert f["bedingungen"]["g"] == {}            # 50 große Telkos < 100 ⇒ alles gelockert (Branche vor Größe)
    assert f["g"]["sigma"] > 0.045                                        # 1-J.-σ aus der gemischten Kohorte


def test_center_drift_moves_the_median_path_to_the_base_rate():
    from domain.factor_model import center_drift
    seg = SegmentConfig(name="A", base_revenue=100.0, forecast_years=7, revenue_growth=D(fixed_value=0.30))
    cfg = SimulationConfig(n_simulations=50, random_seed=1, segments=[seg])
    g0 = np.zeros((50, 7))
    assert np.array_equal(center_drift(g0, 0.30, 0.10, 0.0, 5), g0)
    for lam, want in ((1.0, 0.10), (0.5, np.sqrt(1.3 * 1.1) - 1)):
        sh = {"g": center_drift(g0, 0.30, 0.10, lam, 5), "m": np.zeros((50, 7)), "m_phi": 1.0}
        rev = MonteCarloEngine(cfg, sh).run().segment_cashflows["A"].revenue
        assert np.allclose((rev[:, 4] / 100.0) ** (1 / 5) - 1, want)          # 5-J.-CAGR auf dem Ziel
        assert np.allclose(rev[:, 6] / rev[:, 5], 1.3)                         # danach wieder Analystenwachstum
