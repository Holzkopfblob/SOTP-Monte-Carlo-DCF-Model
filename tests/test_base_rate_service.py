import numpy as np
import pandas as pd
import pytest

from application.base_rate_service import compare, model_paths
from domain.models import DistributionConfig as D, DistributionType as T, SegmentConfig, SimulationConfig
from infrastructure.monte_carlo_engine import MonteCarloEngine


def _results():
    segs = [SegmentConfig(name="A", base_revenue=600.0, forecast_years=6, revenue_growth=D(fixed_value=0.04),
                          ebitda_margin=D(fixed_value=0.30)),
            SegmentConfig(name="B", base_revenue=400.0, forecast_years=5, revenue_growth=D(T.NORMAL, mean=0.04, std=0.01),
                          ebitda_margin=D(fixed_value=0.20))]
    return MonteCarloEngine(SimulationConfig(n_simulations=2000, random_seed=1, segments=segs)).run()


def test_model_paths_constant_growth_and_margin():
    cagr, dm, g1, m1 = model_paths(_results())
    assert np.median(cagr) == pytest.approx(0.04, abs=2e-3) and np.median(g1) == pytest.approx(0.04, abs=2e-3)
    assert np.median(m1) == pytest.approx(0.26, abs=2e-3)              # umsatzgewichtet 0,6·30 % + 0,4·20 %
    assert np.allclose(dm, 0.0, atol=5e-3)                             # Margen konstant ⇒ Mix ändert kaum


def test_compare_against_synthetic_base_rates():
    rng = np.random.default_rng(0)
    o = pd.DataFrame({"sektor": "Telko", "groesse": 1, "g_start": 1, "m_start": 2,
                      "cagr": rng.normal(0.03, 0.05, 5000), "dm": rng.normal(0.0, 0.04, 5000)})
    out = compare(_results(), "Telko", 1.2e9, ob={5: o, 4: o})
    w = out["wachstum_5j"]
    assert w["bedingungen"] == {"sektor": "Telko", "groesse": 1, "g_start": 1, "m_start": 2}
    assert w["perzentil_modell_median"] == pytest.approx(0.58, abs=0.03)    # 4 % in N(3 %, 5 %)
    assert w["breite_basis_zu_modell"] > 5                                  # Modell viel enger als die Außensicht


def test_check_12_flags_position_and_width():
    from domain.checks import OK, WARN, base_rate_checks
    cmp = {"wachstum_5j": {"perzentil_modell_median": 0.97, "breite_basis_zu_modell": 2.0,
                           "basis": {"n": 400}, "bedingungen": {"sektor": "Telko"}},
           "marge_4j": {"perzentil_modell_median": 0.5, "breite_basis_zu_modell": 12.0,
                        "basis": {"n": 400}, "bedingungen": {}}}
    st = {c.name: c.status for c in base_rate_checks(cmp)}
    assert st == {"Basisrate Lage · Umsatz-CAGR 5 J.": WARN, "Basisrate Breite · Umsatz-CAGR 5 J.": OK,
                  "Basisrate Lage · Margenänderung J1→J5": OK, "Basisrate Breite · Margenänderung J1→J5": WARN}
