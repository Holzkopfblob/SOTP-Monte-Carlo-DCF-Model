"""
Tests for application.simulation_service – orchestration & statistics.
"""
from __future__ import annotations

import numpy as np
import pytest

from application.simulation_service import SimulationService
from domain.models import SimulationConfig, SimulationResults


class TestRunSimulation:
    def test_returns_results(self, minimal_sim_config):
        r = SimulationService.run_simulation(minimal_sim_config)
        assert isinstance(r, SimulationResults)
        assert r.n_simulations == minimal_sim_config.n_simulations

    def test_shapes(self, minimal_sim_config):
        n = minimal_sim_config.n_simulations
        r = SimulationService.run_simulation(minimal_sim_config)
        assert r.equity_values.shape == (n,)
        assert r.total_ev.shape == (n,)
        assert r.price_per_share.shape == (n,)
        assert r.pv_corporate_costs.shape == (n,)

    def test_equity_bridge_identity(self, full_sim_config):
        """Equity = Sum(EV) - PV(Corp) - Debt - Minority - Pension + NonOp + Associates."""
        r = SimulationService.run_simulation(full_sim_config)
        bridge = full_sim_config.corporate_bridge
        expected_equity = (
            np.mean(r.total_ev)
            - r.base_corporate_costs_pv
            - bridge.net_debt
            - bridge.minority_interests
            - bridge.pension_liabilities
            + bridge.non_operating_assets
            + bridge.associate_investments
        )
        assert abs(r.base_equity_value - expected_equity) < 0.5

    def test_convergence_populated(self, minimal_sim_config):
        r = SimulationService.run_simulation(minimal_sim_config)
        assert len(r.convergence_indices) > 0
        assert len(r.convergence_means) == len(r.convergence_indices)

    def test_deterministic_with_seed(self, minimal_sim_config):
        r1 = SimulationService.run_simulation(minimal_sim_config)
        r2 = SimulationService.run_simulation(minimal_sim_config)
        np.testing.assert_array_equal(r1.equity_values, r2.equity_values)

    def test_multi_segment_evs(self, full_sim_config):
        r = SimulationService.run_simulation(full_sim_config)
        assert len(r.segment_evs) == 2
        # Sum of segment EVs should equal total EV
        seg_sum = sum(v for v in r.segment_evs.values())
        np.testing.assert_allclose(seg_sum, r.total_ev, rtol=1e-10)


class TestComputeSensitivity:
    def test_returns_sorted_dict(self, full_sim_config):
        r = SimulationService.run_simulation(full_sim_config)
        sens = SimulationService.compute_sensitivity(r)
        assert isinstance(sens, dict)
        # Should be sorted by absolute correlation descending
        abs_vals = [abs(v) for v in sens.values()]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_deterministic_inputs_excluded(self, minimal_sim_config):
        """All-fixed config should yield empty sensitivity (σ ≈ 0)."""
        r = SimulationService.run_simulation(minimal_sim_config)
        sens = SimulationService.compute_sensitivity(r)
        # Minimal config has all-fixed distributions → no stochastic inputs
        assert len(sens) == 0

    def test_correlations_in_range(self, full_sim_config):
        r = SimulationService.run_simulation(full_sim_config)
        sens = SimulationService.compute_sensitivity(r)
        for v in sens.values():
            assert -1.0 <= v <= 1.0


class TestComputeStatistics:
    def test_keys(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = SimulationService.compute_statistics(arr)
        expected_keys = {
            "Mittelwert", "Median", "Std.-Abw.",
            "P5 (5%)", "P25 (25%)", "P75 (75%)", "P95 (95%)",
            "Min", "Max",
            "Schiefe", "Kurtosis", "CV", "IQR",
        }
        assert set(stats.keys()) == expected_keys

    def test_values(self):
        arr = np.arange(1.0, 101.0)
        stats = SimulationService.compute_statistics(arr)
        assert stats["Min"] == 1.0
        assert stats["Max"] == 100.0
        assert abs(stats["Mittelwert"] - 50.5) < 0.01


# ── Tests for _split_bridge_param helper ──────────────────────────────────

from domain.models import DistributionConfig, DistributionType
from presentation.pages.dcf_simulation import _split_bridge_param


class TestSplitBridgeParam:
    def test_none_returns_fallback(self):
        scalar, stoch = _split_bridge_param(None, 42.0)
        assert scalar == 42.0
        assert stoch is None

    def test_fixed_returns_scalar_no_stochastic(self):
        d = DistributionConfig(dist_type=DistributionType.FIXED, fixed_value=99.0)
        scalar, stoch = _split_bridge_param(d)
        assert scalar == 99.0
        assert stoch is None

    def test_normal_returns_stochastic(self):
        d = DistributionConfig(dist_type=DistributionType.NORMAL, mean=50.0, std=5.0)
        scalar, stoch = _split_bridge_param(d)
        assert scalar == 50.0  # representative_value = mean
        assert stoch is d

    def test_pert_returns_stochastic(self):
        d = DistributionConfig(
            dist_type=DistributionType.PERT, low=10.0, mode=20.0, high=30.0,
        )
        scalar, stoch = _split_bridge_param(d)
        assert scalar == 20.0  # representative_value = mode
        assert stoch is d

    def test_uniform_returns_stochastic(self):
        d = DistributionConfig(
            dist_type=DistributionType.UNIFORM, low=40.0, high=60.0,
        )
        scalar, stoch = _split_bridge_param(d)
        assert scalar == 50.0  # representative_value = midpoint
        assert stoch is d
