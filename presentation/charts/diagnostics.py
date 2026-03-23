"""Diagnostics and quality-related charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .common import COLORS, PALETTE_EXTENDED, apply_figure_defaults


def convergence_chart(
    indices: np.ndarray,
    means: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    title: str = "Konvergenz-Diagnose - Laufender Mittelwert (Equity Value)",
) -> go.Figure:
    """Running mean + 95 % CI band showing Monte-Carlo convergence."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([indices, indices[::-1]]),
        y=np.concatenate([ci_high, ci_low[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(width=0),
        name="95 %-KI",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=indices,
        y=means,
        mode="lines",
        name="Laufender Mittelwert",
        line=dict(color=COLORS["primary"], width=2.5),
    ))

    final_mean = means[-1]
    fig.add_hline(
        y=final_mean,
        line_dash="dot",
        line_color=COLORS["neutral"],
        line_width=1,
        annotation_text=f"Endwert: {final_mean:,.1f}",
        annotation_font_size=10,
    )

    final_width = ci_high[-1] - ci_low[-1]
    pct_width = (final_width / abs(final_mean) * 100) if abs(final_mean) > 0 else 0
    fig.add_annotation(
        x=indices[-1],
        y=ci_high[-1],
        text=f"KI-Breite: {final_width:,.1f} ({pct_width:.2f} %)",
        showarrow=True,
        arrowhead=2,
        font=dict(size=11, color=COLORS["secondary"]),
        yshift=15,
    )

    apply_figure_defaults(
        fig,
        title=title,
        xaxis_title="Anzahl Simulationen",
        yaxis_title="Equity Value (Mio.)",
        height=480,
    )
    return fig


def revenue_fade_preview(g_initial: float, g_terminal: float, fade_speed: float, forecast_years: int) -> go.Figure:
    """Preview chart showing the growth rate path under the fade model."""
    years = np.arange(1, forecast_years + 1)
    g_fade = g_terminal + (g_initial - g_terminal) * np.exp(-fade_speed * years)
    g_const = np.full_like(years, g_initial, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=g_fade * 100, mode="lines+markers", name="Fade-Modell", line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=years, y=g_const * 100, mode="lines", name="Konstant", line=dict(color=COLORS["neutral"], width=1.5, dash="dash")))
    fig.add_hline(
        y=g_terminal * 100,
        line_dash="dot",
        line_color=COLORS["secondary"],
        annotation_text=f"Terminal g = {g_terminal*100:.1f} %",
        annotation_font_size=10,
    )

    apply_figure_defaults(
        fig,
        title="Umsatzwachstum über den Prognosezeitraum",
        xaxis_title="Jahr",
        yaxis_title="Wachstumsrate (%)",
        height=350,
        margin_top=50,
    )
    return fig


def parameter_fade_preview(
    fade_speed: float,
    forecast_years: int,
    params: dict[str, tuple[float, float]],
) -> go.Figure:
    """Multi-parameter fade preview chart."""
    years = np.arange(1, forecast_years + 1)
    decay = np.exp(-fade_speed * years)

    fig = go.Figure()
    for label, (p_init, p_term) in params.items():
        path = p_term + (p_init - p_term) * decay
        fig.add_trace(go.Scatter(x=years, y=path, mode="lines+markers", name=label, line=dict(width=2)))

    apply_figure_defaults(
        fig,
        title="Parameter-Fade Vorschau",
        xaxis_title="Jahr",
        yaxis_title="Wert (%)",
        height=350,
        margin_top=50,
    )
    return fig


def tv_ev_decomposition_chart(
    segment_names: list[str],
    mean_pv_fcff_shares: list[float],
    mean_pv_tv_shares: list[float],
) -> go.Figure:
    """Stacked bar: PV(FCFF) vs PV(TV) share per segment."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="PV(TV)",
        x=segment_names,
        y=[v * 100 for v in mean_pv_tv_shares],
        marker_color=COLORS["secondary"],
        text=[f"{v*100:.1f}%" for v in mean_pv_tv_shares],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="PV(FCFF)",
        x=segment_names,
        y=[v * 100 for v in mean_pv_fcff_shares],
        marker_color=COLORS["primary"],
        text=[f"{v*100:.1f}%" for v in mean_pv_fcff_shares],
        textposition="inside",
    ))
    fig.add_hline(y=70, line_dash="dot", line_color=COLORS["negative"], annotation_text="70 % TV-Schwelle", annotation_font_size=10)

    apply_figure_defaults(
        fig,
        title="EV-Zusammensetzung: PV(FCFF) vs. PV(Terminal Value)",
        yaxis_title="Anteil am EV (%)",
        height=420,
    )
    fig.update_layout(barmode="stack", yaxis=dict(range=[0, 105]))
    return fig


def quality_score_gauge(score: dict[str, float]) -> go.Figure:
    """Gauge chart (0-100) for the composite valuation quality score."""
    total = score.get("total", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total,
        number=dict(suffix=" / 100"),
        title=dict(text="Bewertungsqualität"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=COLORS["primary"]),
            steps=[
                dict(range=[0, 40], color="#fee0d2"),
                dict(range=[40, 70], color="#fff3cd"),
                dict(range=[70, 100], color="#d4edda"),
            ],
            threshold=dict(line=dict(color=COLORS["negative"], width=3), thickness=0.8, value=total),
        ),
    ))
    apply_figure_defaults(fig, title="", height=320, margin_top=60, margin_bottom=20)
    return fig


def quality_score_breakdown_chart(score: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of the four quality sub-scores (each 0-25)."""
    labels = ["TV/EV Risiko", "Konvergenz", "Sensitivitäts-\nDiversifikation", "Ergebnis-\nStreuung"]
    keys = ["tv_ev", "convergence", "sensitivity", "dispersion"]
    values = [score.get(k, 0) for k in keys]
    colors = [COLORS["secondary"] if v < 12.5 else COLORS["positive"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} / 25" for v in values],
        textposition="outside",
    ))

    apply_figure_defaults(
        fig,
        title="Qualitäts-Score - Aufschlüsselung",
        xaxis_title="Punkte",
        height=280,
        margin_top=50,
    )
    fig.update_layout(xaxis=dict(range=[0, 28]), margin=dict(l=160, t=50, b=40), showlegend=False)
    return fig


def roic_histogram(segment_roic: dict[str, np.ndarray], wacc_mean: float | None = None) -> go.Figure:
    """Overlaid histograms of implied ROIC per segment with optional WACC line."""
    fig = go.Figure()
    for i, (seg_name, roic_arr) in enumerate(segment_roic.items()):
        fig.add_trace(go.Histogram(
            x=roic_arr * 100,
            nbinsx=60,
            name=seg_name,
            marker_color=PALETTE_EXTENDED[i % len(PALETTE_EXTENDED)],
            opacity=0.6,
            histnorm="probability density",
        ))

    if wacc_mean is not None:
        fig.add_vline(
            x=wacc_mean * 100,
            line_dash="dash",
            line_color=COLORS["negative"],
            line_width=2.5,
            annotation_text=f"Ø WACC: {wacc_mean * 100:.1f} %",
            annotation_font_size=11,
        )

    apply_figure_defaults(
        fig,
        title="Implied ROIC - Verteilung je Segment",
        xaxis_title="Implied ROIC (%)",
        yaxis_title="Dichte",
        height=480,
    )
    fig.update_layout(barmode="overlay")
    return fig


def reinvestment_rate_chart(segment_reinvest: dict[str, np.ndarray]) -> go.Figure:
    """Box plot of reinvestment rates per segment."""
    fig = go.Figure()
    for i, (seg_name, rr_arr) in enumerate(segment_reinvest.items()):
        fig.add_trace(go.Box(
            y=rr_arr * 100,
            name=seg_name,
            marker_color=PALETTE_EXTENDED[i % len(PALETTE_EXTENDED)],
            boxmean="sd",
        ))

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=COLORS["neutral"],
        annotation_text="Null-Reinvestition",
        annotation_font_size=10,
    )

    apply_figure_defaults(
        fig,
        title="Reinvestitionsrate - Verteilung je Segment",
        yaxis_title="Reinvestitionsrate (%)",
        height=440,
        showlegend=False,
    )
    return fig


def roic_vs_wacc_scatter(
    segment_roic: dict[str, np.ndarray],
    segment_wacc: dict[str, np.ndarray],
    max_points: int = 2000,
) -> go.Figure:
    """Scatter plot: Implied ROIC vs. WACC per segment."""
    fig = go.Figure()

    for i, seg_name in enumerate(segment_roic):
        roic_arr = segment_roic[seg_name]
        wacc_arr = segment_wacc[seg_name]
        n = len(roic_arr)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points, dtype=int)
            roic_arr = roic_arr[idx]
            wacc_arr = wacc_arr[idx]

        fig.add_trace(go.Scattergl(
            x=wacc_arr * 100,
            y=roic_arr * 100,
            mode="markers",
            name=seg_name,
            marker=dict(color=PALETTE_EXTENDED[i % len(PALETTE_EXTENDED)], size=3, opacity=0.4),
        ))

    axis_min, axis_max = 0, 50
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="ROIC = WACC",
        line=dict(color=COLORS["neutral"], dash="dash", width=2),
        showlegend=True,
    ))
    fig.add_annotation(x=30, y=40, text="Wertschöpfung ↑", showarrow=False, font=dict(size=12, color=COLORS["positive"]))
    fig.add_annotation(x=30, y=20, text="Wertvernichtung ↓", showarrow=False, font=dict(size=12, color=COLORS["negative"]))

    apply_figure_defaults(
        fig,
        title="ROIC vs. WACC - Wertschöpfungsanalyse",
        xaxis_title="WACC (%)",
        yaxis_title="Implied ROIC (%)",
        height=520,
    )
    return fig


def valuation_confidence_panel(
    *,
    ci_relative_width_pct: float,
    tail_ratio: float | None,
    quality_total: float | None,
) -> go.Figure:
    """Aggregated confidence panel for DCF result robustness."""
    quality_value = float(quality_total) if quality_total is not None else 0.0
    tail_value = float(tail_ratio) if tail_ratio is not None else 0.0

    # Convert CI width to a confidence-like score (lower width = better).
    ci_score = max(0.0, min(100.0, 100.0 - ci_relative_width_pct * 20.0))
    tail_score = max(0.0, min(100.0, tail_value * 100.0))

    labels = ["Konvergenz", "Tail-Stabilität", "Qualitäts-Score"]
    values = [ci_score, tail_score, quality_value]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["positive"]],
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))

    apply_figure_defaults(
        fig,
        title="Valuation Confidence Panel",
        yaxis_title="Score (0-100)",
        height=360,
    )
    fig.update_layout(yaxis=dict(range=[0, 105]), showlegend=False)
    return fig


def portfolio_robustness_panel(method_metrics: list[dict[str, float]]) -> go.Figure:
    """Aggregated robustness panel for portfolio methods."""
    names = [m["name"] for m in method_metrics]
    sharpe = [m["sharpe"] for m in method_metrics]
    cvar = [m["cvar"] for m in method_metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cvar,
        y=sharpe,
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(size=11, color=COLORS["accent"]),
        name="Methoden",
    ))

    apply_figure_defaults(
        fig,
        title="Portfolio Robustness Panel",
        xaxis_title="CVaR (5%)",
        yaxis_title="Sharpe Ratio",
        height=420,
    )
    return fig
