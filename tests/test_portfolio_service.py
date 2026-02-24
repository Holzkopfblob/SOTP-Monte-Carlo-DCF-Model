"""
Tests for application.portfolio_service – analysis, optimisation, stress.
"""
from __future__ import annotations

import numpy as np
import pytest

from application.portfolio_service import (
    AssetInput,
    AssetMetrics,
    PortfolioAnalyser,
    PortfolioResult,
    StressTestResult,
    generate_fv_samples,
)


# ══════════════════════════════════════════════════════════════════════════
# Fair-value sample generation
# ══════════════════════════════════════════════════════════════════════════

class TestGenerateFVSamples:
    @pytest.mark.parametrize("dist_type,params", [
        ("Normal", {"mean": 100, "std": 15}),
        ("Lognormal", {"mean": 100, "std": 15}),
        ("Aus DCF-App (μ, σ, Schiefe)", {"mean": 100, "std": 15, "skew": 0.0}),
        ("Aus DCF-App (μ, σ, Schiefe)", {"mean": 100, "std": 15, "skew": 1.5}),
        ("PERT", {"low": 60, "mode": 100, "high": 140}),
        ("Dreiecksverteilung", {"low": 60, "mode": 100, "high": 140}),
        ("Gleichverteilung", {"low": 60, "high": 140}),
    ])
    def test_shape_and_positive(self, dist_type, params):
        s = generate_fv_samples(dist_type, params, n=1_000, seed=42)
        assert s.shape == (1_000,)
        assert np.all(s >= 0.01)

    def test_unknown_dist(self):
        s = generate_fv_samples("UnknownDist", {"mean": 50.0}, n=10)
        assert s.shape == (10,)

    def test_deterministic(self):
        s1 = generate_fv_samples("Normal", {"mean": 100, "std": 10}, seed=99)
        s2 = generate_fv_samples("Normal", {"mean": 100, "std": 10}, seed=99)
        np.testing.assert_array_equal(s1, s2)


# ══════════════════════════════════════════════════════════════════════════
# Single-asset analysis
# ══════════════════════════════════════════════════════════════════════════

class TestAnalyseAsset:
    def test_metrics_fields(self, sample_assets):
        pa = PortfolioAnalyser(0.03)
        m = pa.analyse_asset(sample_assets[0])
        assert isinstance(m, AssetMetrics)
        assert m.name == "Alpha"
        assert 0.0 <= m.prob_profit <= 1.0
        assert -1.0 <= m.var_5 <= 1.0

    def test_signal_logic(self, sample_assets):
        """Asset with high upside should get 'Kaufen' signal."""
        pa = PortfolioAnalyser(0.03)
        # Alpha: price=100, mean_fv≈120 → MoS ≈ 17%, prob_profit high
        m = pa.analyse_asset(sample_assets[0])
        assert "Kaufen" in m.signal or "Halten" in m.signal

    def test_analyse_all(self, sample_assets):
        pa = PortfolioAnalyser(0.03)
        metrics = pa.analyse_all(sample_assets)
        assert len(metrics) == 3


# ══════════════════════════════════════════════════════════════════════════
# Matrix construction
# ══════════════════════════════════════════════════════════════════════════

class TestMatrixConstruction:
    def test_returns_matrix_shape(self, sample_assets):
        R = PortfolioAnalyser.build_returns_matrix(sample_assets)
        assert R.shape == (5_000, 3)

    def test_sector_correlation_symmetric(self, sample_assets):
        sectors = [a.sector for a in sample_assets]
        corr = PortfolioAnalyser.build_sector_correlation(sectors)
        assert corr.shape == (3, 3)
        np.testing.assert_array_equal(corr, corr.T)
        np.testing.assert_array_equal(np.diag(corr), np.ones(3))

    def test_ensure_psd(self):
        """Ensure PSD projection works on a non-PSD matrix."""
        bad = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ])
        fixed = PortfolioAnalyser.ensure_psd(bad)
        eigvals = np.linalg.eigvalsh(fixed)
        assert np.all(eigvals >= -1e-7)
        np.testing.assert_allclose(np.diag(fixed), 1.0, atol=1e-6)

    def test_build_covariance(self, sample_assets):
        R = PortfolioAnalyser.build_returns_matrix(sample_assets)
        corr = PortfolioAnalyser.build_sector_correlation(
            [a.sector for a in sample_assets]
        )
        mu, stds, cov = PortfolioAnalyser.build_covariance(R, corr)
        assert mu.shape == (3,)
        assert stds.shape == (3,)
        assert cov.shape == (3, 3)


# ══════════════════════════════════════════════════════════════════════════
# Optimisation methods
# ══════════════════════════════════════════════════════════════════════════

class TestOptimisation:
    @pytest.fixture(autouse=True)
    def _setup(self, sample_assets):
        self.pa = PortfolioAnalyser(0.03)
        self.assets = sample_assets
        self.R = self.pa.build_returns_matrix(sample_assets)
        corr = self.pa.build_sector_correlation([a.sector for a in sample_assets])
        self.mu, self.stds, self.cov = self.pa.build_covariance(self.R, corr)
        self.bounds = [(a.min_weight, a.max_weight) for a in sample_assets]
        self.metrics = self.pa.analyse_all(sample_assets)

    def _check_weights(self, result: PortfolioResult | None):
        assert result is not None
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-4)
        assert np.all(result.weights >= -1e-6)

    def test_max_sharpe(self):
        r = self.pa.optimise_max_sharpe(self.mu, self.cov, self.stds, self.R, self.bounds)
        self._check_weights(r)
        assert r.sharpe_ratio > 0

    def test_min_vol(self):
        r = self.pa.optimise_min_vol(self.mu, self.cov, self.stds, self.R, self.bounds)
        self._check_weights(r)

    def test_risk_parity(self):
        r = self.pa.optimise_risk_parity(self.mu, self.cov, self.stds, self.R, self.bounds)
        self._check_weights(r)

    def test_min_cvar(self):
        r = self.pa.optimise_min_cvar(self.mu, self.cov, self.stds, self.R, self.bounds)
        self._check_weights(r)

    def test_max_diversification(self):
        r = self.pa.optimise_max_diversification(self.mu, self.cov, self.stds, self.R, self.bounds)
        self._check_weights(r)

    def test_kelly(self):
        r = self.pa.kelly_weights(self.metrics, self.mu, self.cov, self.stds, self.R)
        self._check_weights(r)

    def test_equal_weights(self):
        r = self.pa.equal_weights(3, self.mu, self.cov, self.stds, self.R)
        self._check_weights(r)
        np.testing.assert_allclose(r.weights, [1/3, 1/3, 1/3])

    def test_run_all_returns_8(self):
        res = self.pa.run_all_optimisations(
            self.metrics, self.mu, self.cov, self.stds, self.R, self.bounds,
        )
        assert len(res) == 8
        # At least equal-weight and kelly should always succeed
        assert res["Gleichgewicht (1/N)"] is not None
        assert res["Kelly (Multi-Asset)"] is not None


# ══════════════════════════════════════════════════════════════════════════
# Efficient frontier
# ══════════════════════════════════════════════════════════════════════════

class TestEfficientFrontier:
    def test_returns_arrays(self, sample_assets):
        pa = PortfolioAnalyser(0.03)
        R = pa.build_returns_matrix(sample_assets)
        corr = pa.build_sector_correlation([a.sector for a in sample_assets])
        mu, _, cov = pa.build_covariance(R, corr)
        vols, rets = pa.efficient_frontier(mu, cov, n_points=20)
        assert len(vols) > 0
        assert len(vols) == len(rets)


# ══════════════════════════════════════════════════════════════════════════
# Stress testing
# ══════════════════════════════════════════════════════════════════════════

class TestStressTest:
    def test_stress_results(self, sample_assets):
        pa = PortfolioAnalyser(0.03)
        R = pa.build_returns_matrix(sample_assets)
        corr = pa.build_sector_correlation([a.sector for a in sample_assets])
        mu, stds, cov = pa.build_covariance(R, corr)
        metrics = pa.analyse_all(sample_assets)
        bounds = [(a.min_weight, a.max_weight) for a in sample_assets]
        opt = pa.run_all_optimisations(metrics, mu, cov, stds, R, bounds)

        portfolios = {k: v.weights for k, v in opt.items() if v is not None}
        results, shocked_returns = pa.stress_test(
            portfolios, R, [a.sector for a in sample_assets],
            market_shock_pct=-30, corr_stress=0.85,
        )
        assert len(results) == len(portfolios)
        assert shocked_returns.shape == R.shape
        for sr in results:
            assert isinstance(sr, StressTestResult)
            assert sr.return_stressed < sr.return_normal  # shock reduces returns

    def test_sector_shock(self, sample_assets):
        pa = PortfolioAnalyser(0.03)
        R = pa.build_returns_matrix(sample_assets)
        w = np.array([1/3, 1/3, 1/3])

        results, shocked = pa.stress_test(
            {"EqW": w}, R, [a.sector for a in sample_assets],
            market_shock_pct=-20, corr_stress=0.0,
            sector_shock="Technologie", sector_shock_pct=-15,
        )
        assert len(results) == 1
        assert shocked.shape == R.shape
