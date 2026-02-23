"""
Tests for domain.portfolio_models – data-class imports and construction.
"""
from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Canonical import path (domain layer)
# ═══════════════════════════════════════════════════════════════════════════

class TestCanonicalImport:
    def test_import_asset_input(self):
        from domain.portfolio_models import AssetInput
        a = AssetInput("X", "Tech", 100.0, np.ones(10))
        assert a.name == "X"

    def test_import_asset_metrics(self):
        from domain.portfolio_models import AssetMetrics
        assert AssetMetrics is not None

    def test_import_portfolio_result(self):
        from domain.portfolio_models import PortfolioResult
        assert PortfolioResult is not None

    def test_import_stress_test_result(self):
        from domain.portfolio_models import StressTestResult
        assert StressTestResult is not None


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible import path (application layer re-exports)
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompatImport:
    def test_reexported_asset_input(self):
        from application.portfolio_service import AssetInput as AI_compat
        from domain.portfolio_models import AssetInput as AI_canon
        assert AI_compat is AI_canon

    def test_reexported_asset_metrics(self):
        from application.portfolio_service import AssetMetrics as AM_compat
        from domain.portfolio_models import AssetMetrics as AM_canon
        assert AM_compat is AM_canon

    def test_reexported_portfolio_result(self):
        from application.portfolio_service import PortfolioResult as PR_compat
        from domain.portfolio_models import PortfolioResult as PR_canon
        assert PR_compat is PR_canon

    def test_reexported_stress_test_result(self):
        from application.portfolio_service import StressTestResult as STR_compat
        from domain.portfolio_models import StressTestResult as STR_canon
        assert STR_compat is STR_canon


# ═══════════════════════════════════════════════════════════════════════════
# AssetInput construction
# ═══════════════════════════════════════════════════════════════════════════

class TestAssetInput:
    def test_default_weight_bounds(self):
        from domain.portfolio_models import AssetInput
        a = AssetInput("A", "Sector", 42.0, np.zeros(5))
        assert a.min_weight == 0.0
        assert a.max_weight == 1.0

    def test_custom_weight_bounds(self):
        from domain.portfolio_models import AssetInput
        a = AssetInput("A", "Sector", 42.0, np.zeros(5), 0.1, 0.5)
        assert a.min_weight == 0.1
        assert a.max_weight == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# PortfolioResult construction
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioResult:
    def test_construction(self):
        from domain.portfolio_models import PortfolioResult
        pr = PortfolioResult(
            name="Test",
            weights=np.array([0.5, 0.5]),
            expected_return=0.08,
            volatility=0.12,
            sharpe_ratio=0.42,
            var_5=-0.10,
            cvar_5=-0.15,
            prob_loss=0.30,
            diversification_ratio=1.2,
            effective_n_assets=1.8,
        )
        assert pr.name == "Test"
        assert pr.weights.sum() == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# StressTestResult construction
# ═══════════════════════════════════════════════════════════════════════════

class TestStressTestResult:
    def test_construction(self):
        from domain.portfolio_models import StressTestResult
        st = StressTestResult(
            method_name="MaxSharpe",
            return_normal=0.10,
            return_stressed=-0.20,
            delta_return=-0.30,
            vol_stressed=0.35,
            var_5_stressed=-0.50,
            cvar_5_stressed=-0.60,
            prob_loss=0.75,
        )
        assert st.method_name == "MaxSharpe"
        assert st.delta_return == pytest.approx(-0.30)
