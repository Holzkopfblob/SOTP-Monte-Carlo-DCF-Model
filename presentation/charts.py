"""
Plotly Chart Generators for the SOTP Monte-Carlo DCF app.

All functions return ``plotly.graph_objects.Figure`` instances that are
ready to be embedded with ``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations


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
    correlations: dict[str, float],
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
    segment_evs: dict[str, float],
    corporate_costs_pv: float,
    net_debt: float,
    equity_value: float,
    *,
    minority_interests: float = 0.0,
    pension_liabilities: float = 0.0,
    non_operating_assets: float = 0.0,
    associate_investments: float = 0.0,
) -> go.Figure:
    """SOTP value-bridge waterfall: Segments − Costs − Debt ± Bridge Items = Equity."""
    names: list[str] = []
    values: list[float] = []
    measures: list[str] = []

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

    # Extended bridge items (only show if non-zero)
    if abs(minority_interests) > 0.01:
        names.append("Minderheitsanteile")
        values.append(-minority_interests)
        measures.append("relative")

    if abs(pension_liabilities) > 0.01:
        names.append("Pensionsr\u00fcckstellungen")
        values.append(-pension_liabilities)
        measures.append("relative")

    if abs(non_operating_assets) > 0.01:
        names.append("Nicht-operative Assets")
        values.append(non_operating_assets)
        measures.append("relative")

    if abs(associate_investments) > 0.01:
        names.append("Beteiligungen")
        values.append(associate_investments)
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
    names: list[str],
    method_weights: dict[str, np.ndarray],
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
    names: list[str],
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


# ═══════════════════════════════════════════════════════════════════════════
# Convergence Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def convergence_chart(
    indices: np.ndarray,
    means: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    title: str = "Konvergenz-Diagnose – Laufender Mittelwert (Equity Value)",
) -> go.Figure:
    """Running mean + 95 % CI band showing Monte-Carlo convergence."""
    fig = go.Figure()

    # Confidence band (shaded area)
    fig.add_trace(go.Scatter(
        x=np.concatenate([indices, indices[::-1]]),
        y=np.concatenate([ci_high, ci_low[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(width=0),
        name="95 %-KI",
        hoverinfo="skip",
    ))

    # Running mean line
    fig.add_trace(go.Scatter(
        x=indices,
        y=means,
        mode="lines",
        name="Laufender Mittelwert",
        line=dict(color=COLORS["primary"], width=2.5),
    ))

    # Final mean reference
    final_mean = means[-1]
    fig.add_hline(
        y=final_mean, line_dash="dot",
        line_color=COLORS["neutral"], line_width=1,
        annotation_text=f"Endwert: {final_mean:,.1f}",
        annotation_font_size=10,
    )

    # CI width annotation: show how narrow the band is at the end
    final_width = ci_high[-1] - ci_low[-1]
    pct_width = (final_width / abs(final_mean) * 100) if abs(final_mean) > 0 else 0

    fig.add_annotation(
        x=indices[-1], y=ci_high[-1],
        text=f"KI-Breite: {final_width:,.1f} ({pct_width:.2f} %)",
        showarrow=True, arrowhead=2,
        font=dict(size=11, color=COLORS["secondary"]),
        yshift=15,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Anzahl Simulationen",
        yaxis_title="Equity Value (Mio.)",
        template=TEMPLATE,
        height=480,
        showlegend=True,
        margin=dict(t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Revenue Fade Preview
# ═══════════════════════════════════════════════════════════════════════════

def revenue_fade_preview(
    g_initial: float,
    g_terminal: float,
    fade_speed: float,
    forecast_years: int,
) -> go.Figure:
    """Preview chart showing the growth rate path under the fade model."""
    years = np.arange(1, forecast_years + 1)
    g_fade = g_terminal + (g_initial - g_terminal) * np.exp(-fade_speed * years)
    g_const = np.full_like(years, g_initial, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=g_fade * 100,
        mode="lines+markers",
        name="Fade-Modell",
        line=dict(color=COLORS["primary"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=g_const * 100,
        mode="lines",
        name="Konstant",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
    ))
    fig.add_hline(
        y=g_terminal * 100,
        line_dash="dot", line_color=COLORS["secondary"],
        annotation_text=f"Terminal g = {g_terminal*100:.1f} %",
        annotation_font_size=10,
    )

    fig.update_layout(
        title="Umsatzwachstum über den Prognosezeitraum",
        xaxis_title="Jahr",
        yaxis_title="Wachstumsrate (%)",
        template=TEMPLATE,
        height=350,
        showlegend=True,
        margin=dict(t=50, b=40),
    )
    return fig


def parameter_fade_preview(
    fade_speed: float,
    forecast_years: int,
    params: dict[str, tuple[float, float]],
) -> go.Figure:
    """Multi-parameter fade preview chart.

    Parameters
    ----------
    fade_speed : float   λ
    forecast_years : int  T
    params : dict
        ``{label: (initial, terminal)}``  – values are in *percent*.
    """
    years = np.arange(1, forecast_years + 1)
    decay = np.exp(-fade_speed * years)

    fig = go.Figure()
    for label, (p_init, p_term) in params.items():
        path = p_term + (p_init - p_term) * decay
        fig.add_trace(go.Scatter(
            x=years, y=path,
            mode="lines+markers",
            name=label,
            line=dict(width=2),
        ))

    fig.update_layout(
        title="Parameter-Fade Vorschau",
        xaxis_title="Jahr",
        yaxis_title="Wert (%)",
        template=TEMPLATE,
        height=350,
        showlegend=True,
        margin=dict(t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 – Core Insights charts
# ═══════════════════════════════════════════════════════════════════════════

def tv_ev_decomposition_chart(
    segment_names: list[str],
    mean_pv_fcff_shares: list[float],
    mean_pv_tv_shares: list[float],
) -> go.Figure:
    """Stacked bar: PV(FCFF) vs PV(TV) share per segment.

    Parameters
    ----------
    mean_pv_fcff_shares : list[float]
        1 − TV/EV  per segment (fraction, 0–1).
    mean_pv_tv_shares : list[float]
        TV/EV  per segment (fraction, 0–1).
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="PV(FCFF)",
        x=segment_names,
        y=[v * 100 for v in mean_pv_fcff_shares],
        marker_color=COLORS["primary"],
        text=[f"{v*100:.1f}%" for v in mean_pv_fcff_shares],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="PV(TV)",
        x=segment_names,
        y=[v * 100 for v in mean_pv_tv_shares],
        marker_color=COLORS["secondary"],
        text=[f"{v*100:.1f}%" for v in mean_pv_tv_shares],
        textposition="inside",
    ))
    fig.add_hline(
        y=70, line_dash="dot", line_color=COLORS["negative"],
        annotation_text="70 % TV-Schwelle",
        annotation_font_size=10,
    )
    fig.update_layout(
        title="EV-Zusammensetzung: PV(FCFF) vs. PV(Terminal Value)",
        yaxis_title="Anteil am EV (%)",
        barmode="stack",
        template=TEMPLATE,
        height=420,
        yaxis=dict(range=[0, 105]),
    )
    return fig


def implied_roic_chart(
    segment_names: list[str],
    roic_means: list[float],
    roic_p5: list[float],
    roic_p95: list[float],
) -> go.Figure:
    """Bar chart of implied ROIC per segment with P5/P95 error bars."""
    fig = go.Figure()
    errors_minus = [max(0, m - lo) for m, lo in zip(roic_means, roic_p5)]
    errors_plus = [max(0, hi - m) for m, hi in zip(roic_means, roic_p95)]

    fig.add_trace(go.Bar(
        x=segment_names,
        y=[v * 100 for v in roic_means],
        marker_color=COLORS["accent"],
        text=[f"{v*100:.0f}%" for v in roic_means],
        textposition="outside",
        error_y=dict(
            type="data",
            symmetric=False,
            array=[v * 100 for v in errors_plus],
            arrayminus=[v * 100 for v in errors_minus],
            color=COLORS["neutral"],
        ),
    ))
    # Common benchmark lines
    fig.add_hline(y=15, line_dash="dot", line_color=COLORS["positive"],
                  annotation_text="15 % (gutes Unternehmen)",
                  annotation_font_size=9, annotation_position="top left")
    fig.add_hline(y=8, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text="8 % (Kapitalkosten-Richtwert)",
                  annotation_font_size=9, annotation_position="bottom left")
    fig.update_layout(
        title="Implizierte ROIC je Segment (Steady-State)",
        yaxis_title="ROIC (%)",
        template=TEMPLATE,
        height=420,
        showlegend=False,
    )
    return fig


def quality_score_gauge(score: dict[str, float]) -> go.Figure:
    """Gauge chart (0–100) for the composite valuation quality score."""
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
            threshold=dict(
                line=dict(color=COLORS["negative"], width=3),
                thickness=0.8,
                value=total,
            ),
        ),
    ))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(t=60, b=20))
    return fig


def quality_score_breakdown_chart(score: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of the four quality sub-scores (each 0–25)."""
    labels = [
        "TV/EV Risiko",
        "Konvergenz",
        "Sensitivitäts-\nDiversifikation",
        "Ergebnis-\nStreuung",
    ]
    keys = ["tv_ev", "convergence", "sensitivity", "dispersion"]
    values = [score.get(k, 0) for k in keys]
    colors = [
        COLORS["secondary"] if v < 12.5 else COLORS["positive"]
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} / 25" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Qualitäts-Score – Aufschlüsselung",
        xaxis_title="Punkte",
        xaxis=dict(range=[0, 28]),
        template=TEMPLATE,
        height=280,
        margin=dict(l=160, t=50, b=40),
        showlegend=False,
    )
    return fig
