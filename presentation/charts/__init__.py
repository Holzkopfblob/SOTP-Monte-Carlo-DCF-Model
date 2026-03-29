"""Chart facade exports for presentation layer."""

from .allocation import waterfall_chart
from .common import COLORS, PALETTE_EXTENDED, TEMPLATE
from .diagnostics import (
    convergence_chart,
    parameter_fade_preview,
    quality_score_breakdown_chart,
    quality_score_gauge,
    reinvestment_rate_chart,
    revenue_fade_preview,
    roic_histogram,
    roic_vs_wacc_scatter,
    tv_ev_decomposition_chart,
    valuation_confidence_panel,
)
from .distribution import cdf_plot, cdf_with_reference, histogram_kde
from .risk import (
    conditional_tornado_chart,
    economic_profit_chart,
    implied_return_cdf,
    margin_of_safety_chart,
    percentile_convergence_chart,
    tornado_chart,
)
from .stress import stress_comparison_chart

__all__ = [
    "COLORS",
    "TEMPLATE",
    "PALETTE_EXTENDED",
    "histogram_kde",
    "cdf_plot",
    "cdf_with_reference",
    "tornado_chart",
    "waterfall_chart",
    "stress_comparison_chart",
    "convergence_chart",
    "revenue_fade_preview",
    "parameter_fade_preview",
    "tv_ev_decomposition_chart",
    "quality_score_gauge",
    "quality_score_breakdown_chart",
    "roic_histogram",
    "reinvestment_rate_chart",
    "roic_vs_wacc_scatter",
    "margin_of_safety_chart",
    "implied_return_cdf",
    "economic_profit_chart",
    "conditional_tornado_chart",
    "percentile_convergence_chart",
    "valuation_confidence_panel",
]
