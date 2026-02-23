"""
Tests for the portfolio-app distribution mapping layer.

Verifies that ``_map_portfolio_dist`` translates UI distribution names
to the correct ``DistributionConfig`` and that ``_to_lognormal_params``
gives correct underlying normal parameters.
"""
from __future__ import annotations

import numpy as np
import pytest

from application.portfolio_service import (
    _map_portfolio_dist,
    _to_lognormal_params,
)
from domain.models import DistributionConfig, DistributionType


# ═══════════════════════════════════════════════════════════════════════════
# _to_lognormal_params
# ═══════════════════════════════════════════════════════════════════════════

class TestToLognormalParams:
    def test_basic_positive(self):
        mu_ln, sigma_ln = _to_lognormal_params(100.0, 20.0)
        # Reconstructed mean should match: exp(mu + sigma^2/2)
        reconstructed = np.exp(mu_ln + sigma_ln ** 2 / 2)
        assert reconstructed == pytest.approx(100.0, rel=1e-6)

    def test_sigma_matches(self):
        mu_ln, sigma_ln = _to_lognormal_params(100.0, 20.0)
        # Reconstructed std: sqrt((exp(sigma^2) - 1)) * exp(2*mu + sigma^2))
        var = (np.exp(sigma_ln ** 2) - 1) * np.exp(2 * mu_ln + sigma_ln ** 2)
        assert np.sqrt(var) == pytest.approx(20.0, rel=1e-6)

    def test_fallback_for_zero_std(self):
        mu_ln, sigma_ln = _to_lognormal_params(100.0, 0.0)
        assert sigma_ln == 0.1  # hardcoded fallback


# ═══════════════════════════════════════════════════════════════════════════
# _map_portfolio_dist
# ═══════════════════════════════════════════════════════════════════════════

class TestMapPortfolioDist:
    def test_normal(self):
        cfg = _map_portfolio_dist("Normal", {"mean": 100, "std": 15})
        assert cfg.dist_type == DistributionType.NORMAL
        assert cfg.mean == 100
        assert cfg.std == 15

    def test_lognormal(self):
        cfg = _map_portfolio_dist("Lognormal", {"mean": 100, "std": 15})
        assert cfg.dist_type == DistributionType.LOGNORMAL
        assert cfg.ln_mu is not None
        assert cfg.ln_sigma is not None

    def test_dcf_app_low_skew_is_normal(self):
        cfg = _map_portfolio_dist(
            "Aus DCF-App (μ, σ, Schiefe)",
            {"mean": 100, "std": 15, "skew": 0.1},
        )
        assert cfg.dist_type == DistributionType.NORMAL

    def test_dcf_app_high_skew_is_lognormal(self):
        cfg = _map_portfolio_dist(
            "Aus DCF-App (μ, σ, Schiefe)",
            {"mean": 100, "std": 15, "skew": 1.5},
        )
        assert cfg.dist_type == DistributionType.LOGNORMAL

    def test_pert(self):
        cfg = _map_portfolio_dist("PERT", {"low": 60, "mode": 100, "high": 140})
        assert cfg.dist_type == DistributionType.PERT
        assert cfg.low == 60
        assert cfg.mode == 100
        assert cfg.high == 140

    def test_triangular(self):
        cfg = _map_portfolio_dist(
            "Dreiecksverteilung", {"low": 60, "mode": 100, "high": 140},
        )
        assert cfg.dist_type == DistributionType.TRIANGULAR

    def test_uniform(self):
        cfg = _map_portfolio_dist("Gleichverteilung", {"low": 60, "high": 140})
        assert cfg.dist_type == DistributionType.UNIFORM
        assert cfg.low == 60
        assert cfg.high == 140

    def test_unknown_fallback_to_fixed(self):
        cfg = _map_portfolio_dist("MadeUpDist", {"mean": 42})
        assert cfg.dist_type == DistributionType.FIXED
        assert cfg.fixed_value == 42
