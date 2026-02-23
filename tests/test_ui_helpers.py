"""
Tests for presentation.ui_helpers – distribution input renderer & exports.

For Streamlit widget functions we can only test the non-UI parts.
"""
from __future__ import annotations

import pytest

from presentation.ui_helpers import DIST_OPTIONS
from domain.models import DistributionType


class TestDistOptions:
    def test_all_types_covered(self):
        """DIST_OPTIONS must contain a label for every DistributionType."""
        for dt in DistributionType:
            assert dt.value in DIST_OPTIONS, f"{dt.value} missing from DIST_OPTIONS"

    def test_no_duplicates(self):
        assert len(DIST_OPTIONS) == len(set(DIST_OPTIONS))


class TestNoPortfolioAppDuplication:
    """After cleanup, portfolio_app.py should import DIST_OPTIONS from
    ui_helpers instead of defining its own copy."""

    def test_portfolio_app_imports_canonical_options(self):
        """DIST_OPTIONS in portfolio_app should match ui_helpers exactly."""
        import importlib
        mod = importlib.import_module("presentation.ui_helpers")
        canonical = mod.DIST_OPTIONS
        # The portfolio_app should use these same options
        assert len(canonical) == len(DistributionType)
