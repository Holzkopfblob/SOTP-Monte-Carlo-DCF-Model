"""Tests for domain.checks – plausibility checks (synthetic numbers only)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from domain.checks import FAIL, INFO, OK, WARN, config_checks, result_checks, roic_is_artifact, year1_fcff
from domain.models import (
    CorporateBridgeConfig, DistributionConfig, RevenueGrowthMode, SegmentConfig, SimulationConfig,
)


def F(v):
    return DistributionConfig(fixed_value=v)


def _cfg(*segs, corp=0.0):
    return SimulationConfig(n_simulations=10, segments=list(segs),
                            corporate_bridge=CorporateBridgeConfig(annual_corporate_costs=corp))


def _by(checks, cid):
    return [c for c in checks if c.id == cid]


class TestCheck4:
    def test_spread_thresholds(self):
        ok = SegmentConfig(name="a", wacc=F(0.07), terminal_growth_rate=F(0.02))
        warn = SegmentConfig(name="b", wacc=F(0.035), terminal_growth_rate=F(0.02))
        fail = SegmentConfig(name="c", wacc=F(0.02), terminal_growth_rate=F(0.02))
        st = [c.status for c in _by(config_checks([ok, warn, fail]), "4")]
        assert st == [OK, WARN, FAIL]


class TestCheck7:
    def test_terminal_da_above_capex_warns(self):
        # Case from the Telekom run on 05.11.2026: D&A terminal 12 % vs. Capex 11 % at g 2 %.
        seg = SegmentConfig(name="us", revenue_growth_mode=RevenueGrowthMode.FADE,
                            capex_pct_revenue=F(0.117), capex_pct_revenue_terminal=F(0.11),
                            da_pct_revenue=F(0.15), da_pct_revenue_terminal=F(0.12),
                            ebitda_margin_terminal=F(0.4), terminal_growth_rate=F(0.02))
        assert _by(config_checks([seg]), "7")[0].status == WARN

    def test_terminal_capex_above_da_ok(self):
        seg = SegmentConfig(name="us", revenue_growth_mode=RevenueGrowthMode.FADE,
                            capex_pct_revenue_terminal=F(0.11), da_pct_revenue_terminal=F(0.105),
                            terminal_growth_rate=F(0.02))
        assert _by(config_checks([seg]), "7")[0].status == OK

    def test_no_growth_no_warning(self):
        seg = SegmentConfig(name="flat", capex_pct_revenue=F(0.05), da_pct_revenue=F(0.06),
                            terminal_growth_rate=F(0.0))
        assert _by(config_checks([seg]), "7")[0].status == OK


class TestRoicArtifact:
    def test_zero_growth_is_artifact(self):
        assert roic_is_artifact(SegmentConfig(revenue_growth=F(0.0)))

    def test_negative_reinvestment_is_artifact(self):
        assert roic_is_artifact(SegmentConfig(revenue_growth=F(0.05), capex_pct_revenue=F(0.02),
                                              da_pct_revenue=F(0.05), nwc_pct_delta_revenue=F(0.0)))

    def test_heavy_capex_with_growth_is_real(self):
        # SpaceX SI-like: capex far above D&A, growth positive → keep the warning.
        assert not roic_is_artifact(SegmentConfig(revenue_growth=F(0.30), capex_pct_revenue=F(2.17),
                                                  da_pct_revenue=F(0.20)))


class TestYear1Fcff:
    def test_constant_growth_by_hand(self):
        seg = SegmentConfig(base_revenue=1000.0, revenue_growth=F(0.10), ebitda_margin=F(0.30),
                            da_pct_revenue=F(0.05), tax_rate=F(0.20), capex_pct_revenue=F(0.06),
                            nwc_pct_delta_revenue=F(0.10))
        # rev 1100; EBIT 275; NOPAT 220; + D&A 55 − Capex 66 − ΔNWC 10 = 199
        assert year1_fcff(seg) == pytest.approx(199.0)

    def test_parameter_fade_applies_in_year_1(self):
        base = dict(base_revenue=1000.0, revenue_growth=F(0.0), da_pct_revenue=F(0.0), tax_rate=F(0.0),
                    capex_pct_revenue=F(0.0), nwc_pct_delta_revenue=F(0.0), fade_speed=0.5,
                    revenue_growth_mode=RevenueGrowthMode.FADE, terminal_growth_rate=F(0.0))
        seg = SegmentConfig(ebitda_margin=F(0.40), ebitda_margin_terminal=F(0.20), **base)
        expected = 1000.0 * (0.20 + 0.20 * np.exp(-0.5))
        assert year1_fcff(seg) == pytest.approx(expected)


def _results(price, tv=0.5, pvd=0.1, name="a"):
    return SimpleNamespace(price_per_share=np.array([price]),
                           segment_tv_ev_ratios={name: np.array([tv])},
                           segment_prob_value_destruction={name: pvd})


class TestResultChecks:
    def test_tv_ev_and_roic_and_fair_value(self):
        seg = SegmentConfig(name="a", revenue_growth=F(0.05))
        ch = result_checks(_cfg(seg), _results(30.0, tv=0.82, pvd=0.9), {"kurs_bei_recherche": 15.0})
        assert _by(ch, "5")[0].status == WARN
        assert _by(ch, "6")[0].status == WARN
        assert _by(ch, "9")[0].status == WARN

    def test_roic_artifact_is_info_not_warning(self):
        seg = SegmentConfig(name="a", revenue_growth=F(0.0))
        ch = result_checks(_cfg(seg), _results(10.0, pvd=0.9))
        assert _by(ch, "6")[0].status == INFO

    def test_roic_artifact_without_warning_stays_ok(self):
        seg = SegmentConfig(name="a", revenue_growth=F(0.0))
        assert _by(result_checks(_cfg(seg), _results(10.0, pvd=0.0)), "6")[0].status == OK

    def test_fcff_vs_ist(self):
        seg = SegmentConfig(base_revenue=1000.0, revenue_growth=F(0.10), ebitda_margin=F(0.30),
                            da_pct_revenue=F(0.05), tax_rate=F(0.20), capex_pct_revenue=F(0.06),
                            nwc_pct_delta_revenue=F(0.10))
        cfg = _cfg(seg, corp=19.0)  # model 199 − 19 = 180
        assert _by(result_checks(cfg, _results(1.0), ist_fcff=175.0), "11")[0].status == OK
        assert _by(result_checks(cfg, _results(1.0), ist_fcff=150.0), "11")[0].status == WARN
        assert _by(result_checks(cfg, _results(1.0), ist_fcff=100.0), "11")[0].status == FAIL
        assert _by(result_checks(cfg, _results(1.0)), "11")[0].status == INFO
