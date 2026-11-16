import numpy as np
import pytest

from domain.irr import R_MAX, equity_at, implied_irr, irr_summary
from domain.models import (
    CorporateBridgeConfig, DistributionConfig as D, DistributionType as T, RevenueGrowthMode,
    SegmentConfig, SimulationConfig, TerminalValueMethod,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine

WACC = D(T.NORMAL, mean=0.08, std=0.01)   # gleiche Verteilung, aber je Segment eigene Ziehung


def _results(corp_costs=0.0):
    gordon_fade = SegmentConfig(
        name="A", base_revenue=1000.0, forecast_years=6, revenue_growth=D(T.PERT, low=0.0, mode=0.06, high=0.12),
        ebitda_margin=D(T.NORMAL, mean=0.30, std=0.03), wacc=WACC,
        terminal_growth_rate=D(T.PERT, low=0.01, mode=0.02, high=0.03),
        revenue_growth_mode=RevenueGrowthMode.FADE, ebitda_margin_terminal=D(T.NORMAL, mean=0.32, std=0.02))
    exit_mult = SegmentConfig(
        name="B", base_revenue=400.0, forecast_years=5, wacc=WACC, terminal_method=TerminalValueMethod.EXIT_MULTIPLE,
        exit_multiple=D(T.NORMAL, mean=9.0, std=1.0), ownership_share=0.6, subsidiary_net_debt=D(fixed_value=150.0))
    bridge = CorporateBridgeConfig(annual_corporate_costs=corp_costs, net_debt=800.0, shares_outstanding=50.0)
    cfg = SimulationConfig(n_simulations=4000, random_seed=7, segments=[gordon_fade, exit_mult], corporate_bridge=bridge)
    return MonteCarloEngine(cfg).run()


def test_rebuilt_equity_matches_engine_at_each_samples_wacc():
    # Ein Segment, keine Holdingkosten: Bewertung bei der gezogenen WACC muss die Engine exakt treffen.
    from domain.irr import segment_ev_at
    r = _results()
    for k in ("A", "B"):
        assert np.allclose(segment_ev_at(r.segment_cashflows[k], r.input_samples[f"{k} | WACC"]), r.segment_evs[k],
                           rtol=1e-10)


def test_ownership_subsidiary_debt_and_exit_multiple_against_engine():
    seg = SegmentConfig(name="B", base_revenue=400.0, forecast_years=5, wacc=WACC,
                        terminal_method=TerminalValueMethod.EXIT_MULTIPLE, exit_multiple=D(T.NORMAL, mean=9.0, std=1.0),
                        ownership_share=0.6, subsidiary_net_debt=D(T.NORMAL, mean=150.0, std=20.0))
    cfg = SimulationConfig(n_simulations=3000, random_seed=5, segments=[seg],
                           corporate_bridge=CorporateBridgeConfig(annual_corporate_costs=0.0, net_debt=200.0,
                                                                  shares_outstanding=30.0))
    r = MonteCarloEngine(cfg).run()
    assert np.allclose(implied_irr(r, r.price_per_share), r.input_samples["B | WACC"], atol=1e-8)


def test_price_equal_to_model_value_returns_that_samples_rate_single_rate_world():
    # Beide Segmente mit derselben WACC-Ziehung ⇒ Preis = Modellwert je Sample ⇒ IRR = diese WACC.
    cfg_seg_wacc = D(T.NORMAL, mean=0.08, std=0.01)
    seg = SegmentConfig(name="A", base_revenue=1000.0, forecast_years=6, wacc=cfg_seg_wacc,
                        revenue_growth=D(T.NORMAL, mean=0.05, std=0.02),
                        terminal_growth_rate=D(T.PERT, low=0.01, mode=0.02, high=0.03))
    cfg = SimulationConfig(n_simulations=3000, random_seed=3, segments=[seg],
                           corporate_bridge=CorporateBridgeConfig(annual_corporate_costs=0.0, net_debt=300.0,
                                                                  shares_outstanding=40.0))
    r = MonteCarloEngine(cfg).run()
    irr = implied_irr(r, r.price_per_share)
    assert np.allclose(irr, r.input_samples["A | WACC"], atol=1e-8)


def test_two_segments_exit_multiple_ownership_and_holding_costs_are_consistent():
    r = _results(corp_costs=40.0)
    price = 45.0
    irr = implied_irr(r, price)
    inside = irr < R_MAX - 1e-6
    # Bei der gefundenen Rate ergibt die Neubewertung genau den Preis (wo eine Wurzel existiert).
    assert np.allclose(equity_at(r, irr)[inside] / r.bridge_samples["aktien"][inside], price, atol=1e-6)
    # Höherer Preis ⇒ niedrigere Rendite, je Sample.
    assert np.all(implied_irr(r, 60.0) <= irr + 1e-12)


def test_summary_fields_and_edge_flags():
    r = _results()
    s = irr_summary(r, 45.0, rf=0.03)
    assert s["p5"] < s["p50"] < s["p95"] and s["vorsprung_p50"] == pytest.approx(s["p50"] - 0.03)
    assert s["anteil_am_rand"] == 0.0
    # Preis weit über jedem endlichen Wert ⇒ Rendite = Untergrenze (größtes ewiges Wachstum + 0,01 pp)
    g = r.segment_cashflows["A"].tv_g
    assert np.allclose(implied_irr(r, 1e9), g + 1e-4, atol=1e-9)
