"""Regression tests for sampling-method functionality in the DCF engine."""

from __future__ import annotations

import pytest

from domain.models import SamplingMethod, SimulationConfig, SegmentConfig, CorporateBridgeConfig
from infrastructure.monte_carlo_engine import MonteCarloEngine


class TestSamplingMethod:
    def test_enum_values(self):
        assert SamplingMethod.PSEUDO_RANDOM.value == "Pseudo-Random (Standard)"
        assert SamplingMethod.SOBOL.value == "Quasi-MC (Sobol)"
        assert not hasattr(SamplingMethod, "ANTITHETIC")

    @pytest.fixture
    def base_config(self) -> SimulationConfig:
        return SimulationConfig(
            n_simulations=256,
            random_seed=42,
            segments=[SegmentConfig(name="S1", base_revenue=1_000)],
            corporate_bridge=CorporateBridgeConfig(net_debt=100, shares_outstanding=100),
        )

    def test_pseudo_random_runs(self, base_config):
        base_config.sampling_method = SamplingMethod.PSEUDO_RANDOM
        engine = MonteCarloEngine(base_config)
        r = engine.run()
        assert r.n_simulations == 256

    def test_sobol_runs(self, base_config):
        base_config.sampling_method = SamplingMethod.SOBOL
        engine = MonteCarloEngine(base_config)
        r = engine.run()
        assert r.n_simulations == 256
