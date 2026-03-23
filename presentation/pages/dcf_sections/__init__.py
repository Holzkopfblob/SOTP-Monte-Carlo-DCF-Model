"""Section renderers for DCF pages."""

from .distribution import render_distribution_section
from .export import render_excel_export_section
from .portfolio_handoff import render_portfolio_handoff_section
from .quality import (
    render_quality_section,
    render_roic_section,
    render_tv_ev_section,
)
from .risk import (
    render_economic_profit_section,
    render_margin_of_safety_section,
    render_tail_risk_section,
)
from .sensitivity import (
    render_conditional_sensitivity_section,
    render_sensitivity_section,
)
from .setup_bridge import render_setup_bridge_section
from .setup_correlation import render_setup_correlation_section
from .setup_simulation import render_setup_simulation_section
from .summary import (
    render_descriptive_stats_section,
    render_key_metrics_section,
)

__all__ = [
    "render_key_metrics_section",
    "render_descriptive_stats_section",
    "render_distribution_section",
    "render_sensitivity_section",
    "render_conditional_sensitivity_section",
    "render_tv_ev_section",
    "render_quality_section",
    "render_roic_section",
    "render_tail_risk_section",
    "render_economic_profit_section",
    "render_margin_of_safety_section",
    "render_portfolio_handoff_section",
    "render_excel_export_section",
    "render_setup_simulation_section",
    "render_setup_bridge_section",
    "render_setup_correlation_section",
]
