"""
Plotly Chart Generators for the SOTP Monte-Carlo DCF app.

All functions return ``plotly.graph_objects.Figure`` instances that are
ready to be embedded with ``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


# ── Colour palette ────────────────────────────────────────────────────────

COLORS = {
    "primary":   "#1f77b4",
    "secondary": "#ff7f0e",
    "positive":  "#2ca02c",
    "negative":  "#d62728",
    "neutral":   "#7f7f7f",
    "accent":    "#9467bd",
}

TEMPLATE = "plotly_white"


# ═══════════════════════════════════════════════════════════════════════════
# Histogram + KDE
# ═══════════════════════════════════════════════════════════════════════════

def histogram_kde(
    values: np.ndarray,
    title: str,
    x_label: str = "Wert",
    n_bins: int = 80,
    color: str = COLORS["primary"],
    vlines: dict | None = None,
    show_percentile_lines: bool = True,
) -> go.Figure:
    """Histogram overlaid with a Kernel Density Estimate curve.

    Parameters
    ----------
    vlines : dict, optional
        Custom vertical reference lines.
        ``{label: (value, colour, dash_style)}``
    show_percentile_lines : bool
        If True, draw P5/P50/P95 lines (default).  Set to False when
        *vlines* already provides the needed annotations.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=n_bins,
        name="Häufigkeit",
        marker_color=color,
        opacity=0.7,
        histnorm="probability density",
    ))

    # KDE overlay
    try:
        kde = gaussian_kde(values, bw_method="scott")
        x_range = np.linspace(
            np.percentile(values, 0.5),
            np.percentile(values, 99.5),
            500,
        )
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode="lines",
            name="KDE",
            line=dict(color=COLORS["secondary"], width=2.5),
        ))
    except Exception:
        pass  # skip KDE if data is degenerate

    # Custom vertical lines
    if vlines:
        for label, (val, clr, dash) in vlines.items():
            fig.add_vline(
                x=val, line_dash=dash, line_color=clr, line_width=2,
                annotation_text=label, annotation_font_size=10,
            )

    # Percentile reference lines
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

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Dichte",
        template=TEMPLATE,
        showlegend=True,
        height=480,
        margin=dict(t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CDF (Cumulative Distribution Function)
# ═══════════════════════════════════════════════════════════════════════════

def cdf_plot(
    values: np.ndarray,
    title: str,
    x_label: str = "Wert",
) -> go.Figure:
    """Empirical CDF with probability reference lines."""
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_vals,
        y=cdf,
        mode="lines",
        name="Empirische CDF",
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Reference lines
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
            x=val, y=prob,
            text=f"{prob:.0%}: {val:,.1f}",
            showarrow=False, yshift=14,
            font=dict(size=10),
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Kumulative Wahrscheinlichkeit",
        template=TEMPLATE,
        height=480,
        yaxis=dict(tickformat=".0%"),
        margin=dict(t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Tornado chart (Sensitivity analysis)
# ═══════════════════════════════════════════════════════════════════════════

def tornado_chart(
    correlations: Dict[str, float],
    title: str = "Sensitivitätsanalyse (Spearman-Rangkorrelation)",
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of Spearman rank correlations with equity value."""
    items = list(correlations.items())[:top_n]
    items.reverse()  # lowest absolute at top → plotly renders bottom-up

    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colours = [
        COLORS["positive"] if v >= 0 else COLORS["negative"]
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Spearman ρ",
        yaxis_title="",
        template=TEMPLATE,
        height=max(400, len(items) * 35 + 120),
        xaxis=dict(range=[-1.05, 1.05]),
        margin=dict(l=280, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SOTP Waterfall chart (value bridge)
# ═══════════════════════════════════════════════════════════════════════════

def waterfall_chart(
    segment_evs: Dict[str, float],
    corporate_costs_pv: float,
    net_debt: float,
    equity_value: float,
) -> go.Figure:
    """SOTP value-bridge waterfall: Segments − Costs − Debt = Equity."""
    names: List[str] = []
    values: List[float] = []
    measures: List[str] = []

    for seg_name, ev in segment_evs.items():
        names.append(f"EV {seg_name}")
        values.append(ev)
        measures.append("relative")

    names.append("Holdingkosten (PV)")
    values.append(-corporate_costs_pv)
    measures.append("relative")

    names.append("Nettoverschuldung")
    values.append(-net_debt)
    measures.append("relative")

    names.append("Equity Value")
    values.append(equity_value)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        name="SOTP Bridge",
        orientation="v",
        measure=measures,
        x=names,
        y=values,
        connector=dict(line=dict(color=COLORS["neutral"])),
        increasing=dict(marker=dict(color=COLORS["positive"])),
        decreasing=dict(marker=dict(color=COLORS["negative"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
        text=[f"{v:,.1f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="SOTP-Wertbrücke (Erwartungswerte / Mio.)",
        yaxis_title="Wert (Mio.)",
        template=TEMPLATE,
        height=520,
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Price-per-share histogram
# ═══════════════════════════════════════════════════════════════════════════

def price_histogram(
    prices: np.ndarray,
    title: str = "Verteilung – Preis je Aktie",
) -> go.Figure:
    """Dedicated histogram for per-share prices."""
    return histogram_kde(
        prices, title=title, x_label="Preis je Aktie",
        color=COLORS["accent"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fair-value vs. current-price overlay
# ═══════════════════════════════════════════════════════════════════════════

def fv_vs_price_chart(
    fair_values: np.ndarray,
    current_price: float,
    asset_name: str,
) -> go.Figure:
    """Histogram of fair values with a vertical line at the current price."""
    fig = histogram_kde(
        fair_values,
        title=f"Fair Value vs. Kurs – {asset_name}",
        x_label="Fair Value / Aktie",
        color=COLORS["primary"],
    )
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color=COLORS["negative"],
        line_width=2.5,
        annotation_text=f"Aktueller Kurs: {current_price:,.2f}",
        annotation_font_size=12,
        annotation_bgcolor="rgba(214,39,40,0.15)",
    )
    # shade profit area
    prob_profit = float(np.mean(fair_values > current_price))
    fig.add_annotation(
        x=float(np.percentile(fair_values, 75)),
        y=0,
        text=f"P(Gewinn) = {prob_profit:.1%}",
        showarrow=False,
        yshift=30,
        font=dict(size=13, color=COLORS["positive"]),
        bgcolor="rgba(44,160,44,0.1)",
    )
    return fig


def return_distribution_chart(
    returns_pct: np.ndarray,
    asset_name: str,
) -> go.Figure:
    """Histogram of expected return (%) with a zero line."""
    fig = histogram_kde(
        returns_pct,
        title=f"Erwartete Rendite – {asset_name}",
        x_label="Rendite (%)",
        color=COLORS["secondary"],
    )
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=COLORS["neutral"],
        line_width=1.5,
        annotation_text="Breakeven",
        annotation_font_size=10,
    )
    return fig


def portfolio_weights_chart(
    names: List[str],
    weights_sharpe: np.ndarray,
    weights_minvol: np.ndarray | None = None,
) -> go.Figure:
    """Grouped bar chart comparing Max-Sharpe and Min-Vol weights."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Max Sharpe",
        x=names,
        y=weights_sharpe * 100,
        marker_color=COLORS["primary"],
        text=[f"{w:.1f}%" for w in weights_sharpe * 100],
        textposition="auto",
    ))

    if weights_minvol is not None:
        fig.add_trace(go.Bar(
            name="Min Volatilität",
            x=names,
            y=weights_minvol * 100,
            marker_color=COLORS["secondary"],
            text=[f"{w:.1f}%" for w in weights_minvol * 100],
            textposition="auto",
        ))

    fig.update_layout(
        title="Optimale Portfolio-Gewichtung",
        yaxis_title="Gewicht (%)",
        template=TEMPLATE,
        barmode="group",
        height=420,
        yaxis=dict(range=[0, 105]),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio-specific charts
# ═══════════════════════════════════════════════════════════════════════════

PALETTE_EXTENDED = [
    COLORS["primary"], COLORS["secondary"], COLORS["positive"],
    COLORS["negative"], COLORS["accent"], COLORS["neutral"],
    "#17becf", "#e377c2",
]


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
        x=sorted_v, y=cdf_y, mode="lines", name="CDF",
        line=dict(color=COLORS["primary"], width=2),
    ))
    if ref_value is not None:
        prob = float(np.mean(values <= ref_value))
        fig.add_vline(
            x=ref_value, line_dash="dash", line_color=COLORS["negative"],
            line_width=2,
            annotation_text=f"{ref_label}: {ref_value:,.2f}",
            annotation_font_size=10,
        )
        fig.add_hline(
            y=prob, line_dash="dot", line_color=COLORS["neutral"],
            annotation_text=f"P(FV ≤ {ref_label}) = {prob:.1%}",
            annotation_font_size=10,
        )
    for pct in [5, 25, 50, 75, 95]:
        val = float(np.percentile(values, pct))
        fig.add_annotation(
            x=val, y=pct / 100, text=f"P{pct}: {val:,.1f}",
            showarrow=True, arrowhead=2, font=dict(size=9),
        )
    fig.update_layout(
        title=title, xaxis_title=x_label,
        yaxis_title="Kumulative Wahrscheinlichkeit",
        template=TEMPLATE, height=440,
        yaxis=dict(tickformat=".0%"),
        margin=dict(t=50, b=40),
    )
    return fig


def portfolio_weights_comparison(
    names: List[str],
    method_weights: Dict[str, np.ndarray],
) -> go.Figure:
    """Grouped bar chart comparing all optimisation methods."""
    fig = go.Figure()
    colors_cycle = PALETTE_EXTENDED
    for i, (method_name, w) in enumerate(method_weights.items()):
        w_pct = np.array(w) * 100
        fig.add_trace(go.Bar(
            name=method_name, x=names, y=w_pct,
            marker_color=colors_cycle[i % len(colors_cycle)],
            text=[f"{v:.1f}%" for v in w_pct],
            textposition="auto",
        ))
    fig.update_layout(
        title="Portfolio-Gewichtungen im Vergleich",
        yaxis_title="Gewicht (%)",
        barmode="group",
        template=TEMPLATE, height=500,
        yaxis=dict(range=[0, 105]),
    )
    return fig


def correlation_heatmap(
    corr_matrix: np.ndarray,
    names: List[str],
) -> go.Figure:
    """Correlation matrix as a coloured heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix, x=names, y=names,
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Korrelationsmatrix",
        template=TEMPLATE, height=max(350, len(names) * 50 + 100),
    )
    return fig


def stress_comparison_chart(
    normal_returns: np.ndarray,
    stressed_returns: np.ndarray,
    method_name: str,
) -> go.Figure:
    """Overlaid histograms comparing normal vs. stressed portfolio returns."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal_returns * 100, nbinsx=80, name="Normal",
        marker_color=COLORS["positive"], opacity=0.5,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=stressed_returns * 100, nbinsx=80, name="Stress",
        marker_color=COLORS["negative"], opacity=0.5,
        histnorm="probability density",
    ))
    fig.add_vline(
        x=0, line_dash="solid", line_color=COLORS["neutral"],
        annotation_text="Breakeven",
    )
    fig.update_layout(
        title=f"Renditeverteilung – {method_name}: Normal vs. Stress",
        xaxis_title="Portfolio-Rendite (%)",
        yaxis_title="Dichte",
        barmode="overlay",
        template=TEMPLATE, height=480,
    )
    return fig
