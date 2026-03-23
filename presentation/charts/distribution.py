"""Distribution-oriented charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from .common import COLORS, apply_figure_defaults


def histogram_kde(
    values: np.ndarray,
    title: str,
    x_label: str = "Wert",
    n_bins: int = 80,
    color: str = COLORS["primary"],
    vlines: dict | None = None,
    show_percentile_lines: bool = True,
) -> go.Figure:
    """Histogram overlaid with a Kernel Density Estimate curve."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=n_bins,
        name="Häufigkeit",
        marker_color=color,
        opacity=0.7,
        histnorm="probability density",
    ))

    try:
        kde = gaussian_kde(values, bw_method="scott")
        x_range = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 500)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode="lines",
            name="KDE",
            line=dict(color=COLORS["secondary"], width=2.5),
        ))
    except Exception:
        pass

    if vlines:
        for label, (val, clr, dash) in vlines.items():
            fig.add_vline(
                x=val,
                line_dash=dash,
                line_color=clr,
                line_width=2,
                annotation_text=label,
                annotation_font_size=10,
            )

    if show_percentile_lines:
        for pct, dash_style in [(5, "dot"), (50, "solid"), (95, "dot")]:
            val = float(np.percentile(values, pct))
            fig.add_vline(
                x=val,
                line_dash=dash_style,
                line_color=COLORS["negative"],
                annotation_text=f"P{pct}: {val:,.1f}",
                annotation_font_size=10,
            )

    apply_figure_defaults(fig, title=title, xaxis_title=x_label, yaxis_title="Dichte", height=480, margin_top=50)
    return fig


def cdf_plot(
    values: np.ndarray,
    title: str,
    x_label: str = "Wert",
    color: str | None = None,
) -> go.Figure:
    """Empirical CDF with probability reference lines."""
    line_color = color or COLORS["primary"]
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_vals,
        y=cdf,
        mode="lines",
        name="Empirische CDF",
        line=dict(color=line_color, width=2),
    ))

    for prob in [0.05, 0.25, 0.50, 0.75, 0.95]:
        val = float(np.percentile(values, prob * 100))
        fig.add_shape(
            type="line", x0=val, x1=val, y0=0, y1=prob,
            line=dict(color=COLORS["neutral"], dash="dot", width=1),
        )
        fig.add_shape(
            type="line",
            x0=float(sorted_vals[0]), x1=val,
            y0=prob, y1=prob,
            line=dict(color=COLORS["neutral"], dash="dot", width=1),
        )
        fig.add_annotation(
            x=val,
            y=prob,
            text=f"{prob:.0%}: {val:,.1f}",
            showarrow=False,
            yshift=14,
            font=dict(size=10),
        )

    apply_figure_defaults(
        fig,
        title=title,
        xaxis_title=x_label,
        yaxis_title="Kumulative Wahrscheinlichkeit",
        height=480,
        margin_top=50,
    )
    fig.update_layout(yaxis=dict(tickformat=".0%"))
    return fig


def cdf_with_reference(
    values: np.ndarray,
    title: str,
    x_label: str = "Wert",
    ref_value: float | None = None,
    ref_label: str = "",
) -> go.Figure:
    """Empirical CDF with percentile annotations and an optional reference line."""
    sorted_v = np.sort(values)
    cdf_y = np.arange(1, len(sorted_v) + 1) / len(sorted_v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_v,
        y=cdf_y,
        mode="lines",
        name="CDF",
        line=dict(color=COLORS["primary"], width=2),
    ))

    if ref_value is not None:
        prob = float(np.mean(values <= ref_value))
        fig.add_vline(
            x=ref_value,
            line_dash="dash",
            line_color=COLORS["negative"],
            line_width=2,
            annotation_text=f"{ref_label}: {ref_value:,.2f}",
            annotation_font_size=10,
        )
        fig.add_hline(
            y=prob,
            line_dash="dot",
            line_color=COLORS["neutral"],
            annotation_text=f"P(FV ≤ {ref_label}) = {prob:.1%}",
            annotation_font_size=10,
        )

    for pct in [5, 25, 50, 75, 95]:
        val = float(np.percentile(values, pct))
        fig.add_annotation(
            x=val,
            y=pct / 100,
            text=f"P{pct}: {val:,.1f}",
            showarrow=True,
            arrowhead=2,
            font=dict(size=9),
        )

    apply_figure_defaults(
        fig,
        title=title,
        xaxis_title=x_label,
        yaxis_title="Kumulative Wahrscheinlichkeit",
        height=440,
        margin_top=50,
    )
    fig.update_layout(yaxis=dict(tickformat=".0%"))
    return fig
