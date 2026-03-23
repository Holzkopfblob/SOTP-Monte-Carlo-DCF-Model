"""Stress-specific charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .common import COLORS, apply_figure_defaults


def stress_comparison_chart(
    normal_returns: np.ndarray,
    stressed_returns: np.ndarray,
    method_name: str,
) -> go.Figure:
    """Overlaid histograms comparing normal vs. stressed portfolio returns."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal_returns * 100,
        nbinsx=80,
        name="Normal",
        marker_color=COLORS["positive"],
        opacity=0.5,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=stressed_returns * 100,
        nbinsx=80,
        name="Stress",
        marker_color=COLORS["negative"],
        opacity=0.5,
        histnorm="probability density",
    ))
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=COLORS["neutral"],
        annotation_text="Breakeven",
    )
    apply_figure_defaults(
        fig,
        title=f"Renditeverteilung - {method_name}: Normal vs. Stress",
        xaxis_title="Portfolio-Rendite (%)",
        yaxis_title="Dichte",
        height=480,
    )
    fig.update_layout(barmode="overlay")
    return fig
