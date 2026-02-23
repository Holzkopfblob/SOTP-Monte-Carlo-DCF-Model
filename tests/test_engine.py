"""
Tests for infrastructure.monte_carlo_engine – the simulation engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    DistributionType,
    SegmentConfig,
    SimulationConfig,
    TerminalValueMethod,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine


class TestMonteCarloEngine:
    def test_run_returns_correct_n(self, minimal_sim_config):
        engine = MonteCarloEngine(minimal_sim_config)
        r = engine.run()
        assert r.n_simulations == minimal_sim_config.n_simulations

    def test_segment_evs_populated(self, minimal_sim_config):
        engine = MonteCarloEngine(minimal_sim_config)
        r = engine.run()
        assert len(r.segment_evs) == 1
        seg_name = minimal_sim_config.segments[0].name
        assert seg_name in r.segment_evs

    def test_stochastic_bridge_samples_tracked(self):
        """When stochastic bridge params are set, they appear in input_samples."""
        cfg = SimulationConfig(
            n_simulations=200,
            random_seed=42,
            segments=[SegmentConfig(name="S1")],
            corporate_bridge=CorporateBridgeConfig(
                stochastic_corporate_costs=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=50.0, std=5.0,
                ),
                stochastic_net_debt=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=500.0, std=50.0,
                ),
            ),
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        assert "Bridge | Holdingkosten" in r.input_samples
        assert "Bridge | Nettoverschuldung" in r.input_samples

    def test_stochastic_discount_rate_tracked(self):
        """When stochastic discount rate is set, it appears in input_samples."""
        cfg = SimulationConfig(
            n_simulations=200,
            random_seed=42,
            segments=[SegmentConfig(name="S1")],
            corporate_bridge=CorporateBridgeConfig(
                stochastic_corporate_cost_discount_rate=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
                ),
            ),
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        assert "Bridge | Diskontierung Holding" in r.input_samples
        # PV should vary because discount rate is stochastic
        assert np.std(r.equity_values) > 0

    def test_wacc_floor(self):
        """WACC should be floored at 0.5%."""
        cfg = SimulationConfig(
            n_simulations=100,
            random_seed=42,
            segments=[SegmentConfig(
                name="NegWACC",
                wacc=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.01, std=0.05,
                ),
            )],
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        # All EVs should be finite (WACC floor prevents division issues)
        assert np.all(np.isfinite(r.total_ev))

    def test_gordon_growth_guard(self):
        """Terminal growth must be < WACC - 0.005."""
        cfg = SimulationConfig(
            n_simulations=100,
            random_seed=42,
            segments=[SegmentConfig(
                name="GordonGuard",
                terminal_growth_rate=DistributionConfig(
                    dist_type=DistributionType.FIXED, fixed_value=0.08,
                ),
                wacc=DistributionConfig(
                    dist_type=DistributionType.FIXED, fixed_value=0.09,
                ),
            )],
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        assert np.all(np.isfinite(r.total_ev))

    def test_exit_multiple_method(self):
        cfg = SimulationConfig(
            n_simulations=100,
            random_seed=42,
            segments=[SegmentConfig(
                name="ExitMult",
                terminal_method=TerminalValueMethod.EXIT_MULTIPLE,
                exit_multiple=DistributionConfig(
                    dist_type=DistributionType.FIXED, fixed_value=12.0,
                ),
            )],
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        assert r.total_ev.shape == (100,)
        assert np.all(r.total_ev > 0)

    def test_convergence_diagnostics(self, minimal_sim_config):
        engine = MonteCarloEngine(minimal_sim_config)
        r = engine.run()
        assert len(r.convergence_indices) > 0
        assert r.convergence_means.shape == r.convergence_indices.shape
        # CI should get narrower
        widths = r.convergence_ci_high - r.convergence_ci_low
        assert widths[-1] <= widths[0] + 1e-6  # last should be <= first (or close)


class TestExcelExport:
    def test_generates_bytes(self, minimal_sim_config):
        from infrastructure.excel_export import ExcelExporter
        engine = MonteCarloEngine(minimal_sim_config)
        r = engine.run()
        exporter = ExcelExporter(minimal_sim_config, r)
        data = exporter.generate()
        assert isinstance(data, bytes)
        assert len(data) > 0
        # XLSX magic bytes (PK zip)
        assert data[:2] == b"PK"
