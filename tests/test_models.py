"""
Tests for domain.models – data structures, enums, defaults.

Checks that model defaults are sensible and serialisation-friendly.
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    DistributionType,
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    SimulationResults,
    TerminalValueMethod,
)


class TestDistributionConfig:
    def test_default_is_fixed(self):
        dc = DistributionConfig()
        assert dc.dist_type == DistributionType.FIXED

    def test_all_types_have_string_value(self):
        for dt in DistributionType:
            assert isinstance(dt.value, str)
            assert len(dt.value) > 0

    @pytest.mark.parametrize("dist_type, kwargs, expected", [
        (DistributionType.FIXED,       {"fixed_value": 0.07},               0.07),
        (DistributionType.NORMAL,      {"mean": 0.05, "std": 0.01},        0.05),
        (DistributionType.LOGNORMAL,   {"ln_mu": 0.06, "ln_sigma": 0.02},  0.06),
        (DistributionType.TRIANGULAR,  {"low": 0.02, "mode": 0.05, "high": 0.08}, 0.05),
        (DistributionType.PERT,        {"low": 0.01, "mode": 0.03, "high": 0.06}, 0.03),
        (DistributionType.UNIFORM,     {"low": 0.02, "high": 0.08},        0.05),
    ])
    def test_representative_value(self, dist_type, kwargs, expected):
        dc = DistributionConfig(dist_type=dist_type, **kwargs)
        assert dc.representative_value() == pytest.approx(expected, abs=1e-9)


class TestSegmentConfig:
    def test_defaults(self):
        sc = SegmentConfig()
        assert sc.name == "Segment"
        assert sc.base_revenue == 1_000.0
        assert sc.forecast_years == 5
        assert sc.revenue_growth_mode == RevenueGrowthMode.CONSTANT

    def test_fade_mode(self):
        sc = SegmentConfig(revenue_growth_mode=RevenueGrowthMode.FADE, fade_speed=0.8)
        assert sc.revenue_growth_mode == RevenueGrowthMode.FADE
        assert sc.fade_speed == 0.8


class TestCorporateBridgeConfig:
    def test_defaults(self):
        b = CorporateBridgeConfig()
        assert b.minority_interests == 0.0
        assert b.stochastic_corporate_costs is None

    def test_extended_bridge_items(self):
        b = CorporateBridgeConfig(
            minority_interests=50.0,
            pension_liabilities=80.0,
            non_operating_assets=100.0,
            associate_investments=30.0,
        )
        assert b.minority_interests == 50.0
        assert b.pension_liabilities == 80.0


class TestSimulationConfig:
    def test_defaults(self):
        c = SimulationConfig()
        assert c.n_simulations == 10_000
        assert c.mid_year_convention is True
        assert len(c.segments) == 0


class TestSimulationResults:
    def test_creation(self):
        r = SimulationResults(
            equity_values=np.array([100.0, 200.0]),
            total_ev=np.array([150.0, 250.0]),
            segment_evs={"A": np.array([150.0, 250.0])},
            pv_corporate_costs=np.array([10.0, 10.0]),
            input_samples={},
            n_simulations=2,
        )
        assert r.n_simulations == 2
        assert r.base_equity_value == 0.0  # default
