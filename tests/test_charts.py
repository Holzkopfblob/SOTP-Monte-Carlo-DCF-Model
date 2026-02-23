"""
Tests for presentation.charts – chart functions return valid Plotly figures.

We do NOT test visual correctness (that's Plotly's job), only that:
- Functions return go.Figure
- No exceptions on valid input
- Dead-code functions are confirmed removed after cleanup
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from presentation.charts import (
    cdf_plot,
    cdf_with_reference,
    convergence_chart,
    correlation_heatmap,
    histogram_kde,
    portfolio_weights_comparison,
    revenue_fade_preview,
    stress_comparison_chart,
    tornado_chart,
    waterfall_chart,
)


@pytest.fixture
def sample_values():
    return np.random.default_rng(42).normal(100, 15, 5_000)


class TestHistogramKDE:
    def test_returns_figure(self, sample_values):
        fig = histogram_kde(sample_values, "Test", "X")
        assert isinstance(fig, go.Figure)

    def test_custom_vlines(self, sample_values):
        vlines = {"Ref": (100.0, "#ff0000", "dash")}
        fig = histogram_kde(sample_values, "Test", "X", vlines=vlines)
        assert isinstance(fig, go.Figure)

    def test_no_percentile_lines(self, sample_values):
        fig = histogram_kde(sample_values, "T", "X", show_percentile_lines=False)
        assert isinstance(fig, go.Figure)


class TestCDFPlot:
    def test_returns_figure(self, sample_values):
        fig = cdf_plot(sample_values, "CDF Test", "X")
        assert isinstance(fig, go.Figure)


class TestTornadoChart:
    def test_returns_figure(self):
        corr = {"Param A": 0.8, "Param B": -0.5, "Param C": 0.3}
        fig = tornado_chart(corr)
        assert isinstance(fig, go.Figure)

    def test_empty_dict(self):
        fig = tornado_chart({})
        assert isinstance(fig, go.Figure)


class TestWaterfallChart:
    def test_basic(self):
        fig = waterfall_chart(
            {"Seg A": 500.0, "Seg B": 300.0},
            corporate_costs_pv=55.6,
            net_debt=200.0,
            equity_value=544.4,
        )
        assert isinstance(fig, go.Figure)

    def test_extended_bridge(self):
        fig = waterfall_chart(
            {"S1": 1000.0},
            corporate_costs_pv=100.0,
            net_debt=300.0,
            equity_value=450.0,
            minority_interests=50.0,
            pension_liabilities=30.0,
            non_operating_assets=20.0,
            associate_investments=10.0,
        )
        assert isinstance(fig, go.Figure)


class TestConvergenceChart:
    def test_returns_figure(self):
        idx = np.arange(10, 110, 10)
        means = np.linspace(100, 105, 10)
        lo = means - 5
        hi = means + 5
        fig = convergence_chart(idx, means, lo, hi)
        assert isinstance(fig, go.Figure)


class TestRevenueFadePreview:
    def test_returns_figure(self):
        fig = revenue_fade_preview(0.15, 0.02, 0.5, 7)
        assert isinstance(fig, go.Figure)


class TestCDFWithReference:
    def test_with_ref(self, sample_values):
        fig = cdf_with_reference(sample_values, "CDF", "X", ref_value=100.0, ref_label="Kurs")
        assert isinstance(fig, go.Figure)


class TestPortfolioWeightsComparison:
    def test_returns_figure(self):
        fig = portfolio_weights_comparison(
            ["A", "B"], {"EqW": np.array([0.5, 0.5]), "MS": np.array([0.7, 0.3])},
        )
        assert isinstance(fig, go.Figure)


class TestCorrelationHeatmap:
    def test_returns_figure(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        fig = correlation_heatmap(corr, ["A", "B"])
        assert isinstance(fig, go.Figure)


class TestStressComparisonChart:
    def test_returns_figure(self):
        normal = np.random.default_rng(1).normal(0.05, 0.1, 1000)
        stressed = np.random.default_rng(2).normal(-0.15, 0.15, 1000)
        fig = stress_comparison_chart(normal, stressed, "Test Method")
        assert isinstance(fig, go.Figure)


class TestDeadCodeRemoved:
    """Verify dead-code wrappers have been removed."""

    def test_fv_vs_price_removed(self):
        from presentation import charts
        assert not hasattr(charts, "fv_vs_price_chart")

    def test_return_distribution_chart_removed(self):
        from presentation import charts
        assert not hasattr(charts, "return_distribution_chart")

    def test_price_histogram_removed(self):
        from presentation import charts
        assert not hasattr(charts, "price_histogram"), (
            "price_histogram was a thin wrapper – use histogram_kde directly"
        )
