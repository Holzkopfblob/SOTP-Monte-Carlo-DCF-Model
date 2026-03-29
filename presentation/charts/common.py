"""Common chart constants and layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from presentation.theme.tokens import CHART_COLORS, PLOTLY_TEMPLATE

COLORS = {
    "primary": CHART_COLORS["primary"],
    "secondary": CHART_COLORS["secondary"],
    "positive": CHART_COLORS["positive"],
    "negative": CHART_COLORS["negative"],
    "neutral": CHART_COLORS["neutral"],
    "accent": CHART_COLORS["accent"],
}

TEMPLATE = PLOTLY_TEMPLATE

PALETTE_EXTENDED = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["positive"],
    COLORS["negative"],
    COLORS["accent"],
    COLORS["neutral"],
    CHART_COLORS["tertiary"],
    CHART_COLORS["quaternary"],
]


def apply_figure_defaults(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 480,
    margin_top: int = 60,
    margin_bottom: int = 40,
    showlegend: bool | None = None,
) -> go.Figure:
    """Apply standard figure defaults for consistent appearance."""
    update_kwargs: dict = {
        "title": title,
        "template": TEMPLATE,
        "height": height,
        "margin": dict(t=margin_top, b=margin_bottom),
    }
    if xaxis_title is not None:
        update_kwargs["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        update_kwargs["yaxis_title"] = yaxis_title
    if showlegend is not None:
        update_kwargs["showlegend"] = showlegend

    fig.update_layout(**update_kwargs)
    return fig
