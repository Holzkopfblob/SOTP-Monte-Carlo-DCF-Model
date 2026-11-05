"""Tests for Punkt 4: ownership share and subsidiary net debt per segment."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from domain.checks import result_checks
from domain.models import (
    CorporateBridgeConfig, DistributionConfig, DistributionType, SegmentConfig, SimulationConfig,
)
from infrastructure.config_io import apply_config, collect_config, segment_config_from_state
from infrastructure.monte_carlo_engine import MonteCarloEngine


def F(v):
    return DistributionConfig(fixed_value=v)


def _seg(name, rev, **kw):
    return SegmentConfig(name=name, base_revenue=rev, wacc=DistributionConfig(
        dist_type=DistributionType.PERT, low=0.07, mode=0.08, high=0.10), **kw)


def _cfg(*segs, nd=500.0, n=300):
    return SimulationConfig(n_simulations=n, random_seed=3, segments=list(segs),
                            corporate_bridge=CorporateBridgeConfig(annual_corporate_costs=10.0, net_debt=nd,
                                                                   shares_outstanding=10.0))


def _run(cfg):
    return MonteCarloEngine(cfg).run()


class TestEngine:
    def test_default_share_is_bit_identical(self):
        plain = _run(_cfg(_seg("A", 1000.0), _seg("B", 500.0)))
        explicit = _run(_cfg(_seg("A", 1000.0, ownership_share=1.0, subsidiary_net_debt=F(999.0)),
                             _seg("B", 500.0)))
        assert np.array_equal(plain.price_per_share, explicit.price_per_share)
        assert np.array_equal(plain.total_ev, explicit.total_ev)

    def test_formula(self):
        full = _run(_cfg(_seg("A", 1000.0), _seg("B", 500.0)))
        part = _run(_cfg(_seg("A", 1000.0, ownership_share=0.6, subsidiary_net_debt=F(100.0)), _seg("B", 500.0)))
        ev_a, ev_b = full.segment_evs["A"], full.segment_evs["B"]
        expected = 0.6 * ev_a + ev_b - full.pv_corporate_costs - (500.0 - 0.4 * 100.0)
        assert np.allclose(part.equity_values, expected, rtol=1e-12)
        assert np.allclose(part.total_ev, 0.6 * ev_a + ev_b, rtol=1e-12)
        assert np.array_equal(part.segment_evs["A"], ev_a)  # segment EV stays at 100 %

    def test_share_equals_scaled_revenue(self):
        a = _run(_cfg(_seg("A", 1000.0, ownership_share=0.548)))
        b = _run(_cfg(_seg("A", 548.0)))
        assert np.allclose(a.price_per_share, b.price_per_share, rtol=1e-12)

    def test_waterfall_adds_up(self):
        r = _run(_cfg(_seg("A", 1000.0, ownership_share=0.6, subsidiary_net_debt=F(100.0)), _seg("B", 500.0)))
        bars = (sum(r.base_segment_evs.values()) - r.base_subsidiary_minorities
                - r.base_corporate_costs_pv - r.base_net_debt)
        assert bars == pytest.approx(r.base_equity_value, rel=1e-12)
        assert r.base_net_debt == pytest.approx(500.0) and r.segment_ownership == {"A": 0.6, "B": 1.0}

    def test_full_ownership_ignores_subsidiary_debt_entirely(self):
        snd = DistributionConfig(dist_type=DistributionType.PERT, low=80.0, mode=100.0, high=130.0)
        plain = _run(_cfg(_seg("A", 1000.0)))
        r = _run(_cfg(_seg("A", 1000.0, ownership_share=1.0, subsidiary_net_debt=snd)))
        assert np.array_equal(plain.price_per_share, r.price_per_share)
        assert "A | Nettoschulden Tochter" not in r.input_samples

    def test_stochastic_subsidiary_debt_is_a_driver(self):
        snd = DistributionConfig(dist_type=DistributionType.PERT, low=80.0, mode=100.0, high=130.0)
        r = _run(_cfg(_seg("A", 1000.0, ownership_share=0.5, subsidiary_net_debt=snd)))
        assert "A | Nettoschulden Tochter" in r.input_samples


class TestConfigAndChecks:
    def test_round_trip_and_builder(self):
        state = {"setup_n_seg": 1, "seg_0_name": "US", "seg_0_basrev": 80719.0, "seg_0_anteil": 54.8,
                 "s0_snd_dtype": "Fest (Deterministisch)", "s0_snd_fixed": 72206.0}
        again = apply_config(collect_config(state), {})
        assert again["seg_0_anteil"] == 54.8 and again["s0_snd_fixed"] == 72206.0
        seg = segment_config_from_state(again, 0)
        assert seg.ownership_share == pytest.approx(0.548)
        assert seg.subsidiary_net_debt.fixed_value == pytest.approx(72206.0)  # Mio., not %

    def test_missing_fields_default_to_full_ownership(self):
        seg = segment_config_from_state({"seg_0_basrev": 10.0}, 0)
        assert seg.ownership_share == 1.0 and seg.subsidiary_net_debt.fixed_value == 0.0

    def test_check11_is_attributable(self):
        seg = SegmentConfig(name="A", base_revenue=1000.0, revenue_growth=F(0.10), ebitda_margin=F(0.30),
                            da_pct_revenue=F(0.05), tax_rate=F(0.20), capex_pct_revenue=F(0.06),
                            nwc_pct_delta_revenue=F(0.10), ownership_share=0.5)   # 199 × 0.5
        res = SimpleNamespace(price_per_share=np.array([1.0]), segment_tv_ev_ratios={},
                              segment_prob_value_destruction={})
        cfg = SimulationConfig(segments=[seg], corporate_bridge=CorporateBridgeConfig(annual_corporate_costs=0.0))
        c11 = [c for c in result_checks(cfg, res, ist_fcff=99.5) if c.id == "11"][0]
        assert c11.detail.startswith("Modell 100 ") and c11.status == "✅"   # 199 × 0,5, not 199
