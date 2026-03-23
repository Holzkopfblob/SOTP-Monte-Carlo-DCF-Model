"""Risk and sensitivity-related charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .common import COLORS, PALETTE_EXTENDED, apply_figure_defaults


def tornado_chart(
    correlations: dict[str, float],
    title: str = "Sensitivitätsanalyse (Spearman-Rangkorrelation)",
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of Spearman rank correlations."""
    items = list(correlations.items())[:top_n]
    items.reverse()
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colours = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))

    apply_figure_defaults(
        fig,
        title=title,
        xaxis_title="Spearman ρ",
        yaxis_title="",
        height=max(400, len(items) * 35 + 120),
    )
    fig.update_layout(xaxis=dict(range=[-1.05, 1.05]), margin=dict(l=280, t=50, b=40))
    return fig


def margin_of_safety_chart(price_per_share: np.ndarray, market_price: float) -> go.Figure:
    """Histogram showing upside/downside with market price reference."""
    fig = go.Figure()
    upside = price_per_share[price_per_share >= market_price]
    downside = price_per_share[price_per_share < market_price]

    fig.add_trace(go.Histogram(
        x=downside,
        nbinsx=50,
        name="Downside (FV < Kurs)",
        marker_color=COLORS["negative"],
        opacity=0.6,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=upside,
        nbinsx=50,
        name="Upside (FV ≥ Kurs)",
        marker_color=COLORS["positive"],
        opacity=0.6,
        histnorm="probability density",
    ))

    fig.add_vline(
        x=market_price,
        line_dash="solid",
        line_color=COLORS["neutral"],
        line_width=3,
        annotation_text=f"Marktpreis: {market_price:,.2f}",
        annotation_font_size=12,
    )
    p_upside = float(np.mean(price_per_share >= market_price))
    fig.add_annotation(
        x=float(np.percentile(price_per_share, 75)),
        yref="paper",
        y=0.95,
        text=f"P(Upside) = {p_upside:.1%}",
        showarrow=False,
        font=dict(size=14, color=COLORS["positive"]),
    )

    apply_figure_defaults(
        fig,
        title="Margin-of-Safety Analyse - Fair Value vs. Marktpreis",
        xaxis_title="Fair Value je Aktie",
        yaxis_title="Dichte",
        height=480,
    )
    fig.update_layout(barmode="stack")
    return fig


def implied_return_cdf(price_per_share: np.ndarray, market_price: float) -> go.Figure:
    """CDF of implied return (FV/Price - 1) with key probability markers."""
    returns = (price_per_share / market_price - 1.0) * 100
    sorted_r = np.sort(returns)
    cdf_y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_r,
        y=cdf_y,
        mode="lines",
        name="Implizierte Rendite",
        line=dict(color=COLORS["primary"], width=2),
    ))

    fig.add_vline(x=0, line_dash="solid", line_color=COLORS["neutral"], line_width=2, annotation_text="Breakeven")
    for pct in [5, 25, 50, 75, 95]:
        val = float(np.percentile(returns, pct))
        fig.add_annotation(
            x=val,
            y=pct / 100,
            text=f"P{pct}: {val:+.1f}%",
            showarrow=True,
            arrowhead=2,
            font=dict(size=9),
        )

    apply_figure_defaults(
        fig,
        title="CDF - Implizierte Rendite (Fair Value vs. Marktpreis)",
        xaxis_title="Implizierte Rendite (%)",
        yaxis_title="Kumulative Wahrscheinlichkeit",
        height=480,
    )
    fig.update_layout(yaxis=dict(tickformat=".0%"))
    return fig


def economic_profit_chart(segment_ep: dict[str, np.ndarray]) -> go.Figure:
    """Box plot of economic profit per segment."""
    fig = go.Figure()
    for i, (seg_name, ep_arr) in enumerate(segment_ep.items()):
        fig.add_trace(go.Box(
            y=ep_arr,
            name=seg_name,
            marker_color=PALETTE_EXTENDED[i % len(PALETTE_EXTENDED)],
            boxmean="sd",
        ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["negative"],
        line_width=2,
        annotation_text="EP = 0 (wertschöpfungsneutral)",
        annotation_font_size=10,
    )

    apply_figure_defaults(
        fig,
        title="Economic Profit (EVA) - Verteilung je Segment",
        yaxis_title="Economic Profit (Mio.)",
        height=440,
        showlegend=False,
    )
    return fig


def conditional_tornado_chart(
    bear_corr: dict[str, float],
    bull_corr: dict[str, float],
    top_n: int = 10,
) -> go.Figure:
    """Side-by-side tornado: what drives value in bear vs bull scenarios."""
    all_keys = set(list(bear_corr.keys())[:top_n]) | set(list(bull_corr.keys())[:top_n])
    if not all_keys:
        return go.Figure()

    sorted_keys = sorted(
        all_keys,
        key=lambda k: max(abs(bear_corr.get(k, 0)), abs(bull_corr.get(k, 0))),
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[bear_corr.get(k, 0) for k in sorted_keys],
        orientation="h",
        name="Bear (P<25%)",
        marker_color=COLORS["negative"],
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[bull_corr.get(k, 0) for k in sorted_keys],
        orientation="h",
        name="Bull (P>75%)",
        marker_color=COLORS["positive"],
        opacity=0.7,
    ))

    apply_figure_defaults(
        fig,
        title="Conditional Sensitivity - Bear vs. Bull",
        xaxis_title="Spearman ρ",
        height=max(400, len(sorted_keys) * 40 + 120),
    )
    fig.update_layout(barmode="group", xaxis=dict(range=[-1.05, 1.05]), margin=dict(l=280, t=60, b=40))
    return fig


def percentile_convergence_chart(
    indices: np.ndarray,
    p5: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
) -> go.Figure:
    """Running percentiles (P5/P50/P95) showing tail stability."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=p95, mode="lines", name="P95", line=dict(color=COLORS["positive"], width=2)))
    fig.add_trace(go.Scatter(x=indices, y=p50, mode="lines", name="P50 (Median)", line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=indices, y=p5, mode="lines", name="P5", line=dict(color=COLORS["negative"], width=2)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([indices, indices[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.08)",
        line=dict(width=0),
        name="P5-P95 Band",
        hoverinfo="skip",
    ))

    apply_figure_defaults(
        fig,
        title="Perzentil-Konvergenz - Tail-Stabilität",
        xaxis_title="Anzahl Simulationen",
        yaxis_title="Equity Value (Mio.)",
        height=480,
    )
    return fig
