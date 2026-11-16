"""Diagramme der Portfolio-Ansicht (Plotly, Farben aus presentation.theme).

Farbregeln (dataviz-Skill, validiert): Ist = secondary (orange), Ziel = primary (blau); das Paar besteht die
CVD-Prüfung, Orange liegt unter 3:1 Kontrast ⇒ Werte immer zusätzlich als Beschriftung/Tabelle.
Polarität (Kauf/Verkauf, Upside +/−, Korrelation) = dieselben zwei Pole, Mitte neutral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from presentation.charts.common import COLORS, TEMPLATE, hex_to_rgba

IST, ZIEL = COLORS["secondary"], COLORS["primary"]
POS, NEG, MID = COLORS["primary"], COLORS["secondary"], "#eef0f4"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1f2633")
MUTED = "#5b6475"


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, font=FONT, margin=dict(l=8, r=16, t=36, b=8),
                      plot_bgcolor="white", paper_bgcolor="white", hoverlabel=dict(font_size=13), **kw)
    fig.update_xaxes(gridcolor="#eceff3", zeroline=False, linecolor="#d5dae2")
    fig.update_yaxes(gridcolor="#eceff3", zeroline=False, linecolor="#d5dae2")
    return fig


def allocation_chart(t: pd.DataFrame) -> go.Figure:
    """Ist- vs. Ziel-Gewicht je Titel (horizontal, nach Ziel sortiert), Cash zuletzt."""
    d = t[(t["Ist-Gewicht"].abs() > 1e-4) | (t["Ziel-Gewicht"].abs() > 1e-4)].copy()
    cash = d[d["Firma"] == "Cash (EUR)"]
    # Plotly zeichnet die erste Kategorie unten: Cash unten, darüber aufsteigend nach Ziel-Gewicht
    d = pd.concat([cash, d[d["Firma"] != "Cash (EUR)"].sort_values("Ziel-Gewicht")])
    names = [n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "") for n in d["Firma"]]
    fig = go.Figure()
    for col, color, label in (("Ist-Gewicht", IST, "Ist"), ("Ziel-Gewicht", ZIEL, "Ziel")):
        fig.add_bar(y=names, x=d[col], name=label, orientation="h", marker=dict(color=color, cornerradius=4),
                    text=[f"{v:.0%}" if label == "Ziel" and v > 0.004 else "" for v in d[col]],
                    textposition="outside", textfont=dict(color=MUTED, size=12), cliponaxis=False,
                    hovertemplate="%{y}<br>" + label + ": %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12,
                      legend=dict(orientation="h", y=1.08, x=0, title=None))
    fig.update_xaxes(tickformat=".0%", title=None, rangemode="tozero")
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=names)
    return _layout(fig, max(320, 34 * len(names) + 80))


def wealth_chart(samples: dict, stats_ist: dict, stats_ziel: dict, horizon: int) -> go.Figure:
    """Verteilung des Depotwerts in *horizon* Jahren: Ist vs. Ziel als Dichtekurven, P5/P50/P95 markiert."""
    lo = min(np.percentile(v, 0.5) for v in samples.values())
    hi = max(np.percentile(v, 99.5) for v in samples.values())
    grid = np.linspace(lo, hi, 160)
    fig = go.Figure()
    for key, color in (("Ist", IST), ("Ziel", ZIEL)):
        dens, edges = np.histogram(samples[key], bins=grid, density=True)
        x = (edges[:-1] + edges[1:]) / 2
        dens = np.convolve(dens, np.ones(5) / 5, mode="same")
        fig.add_scatter(x=x, y=dens, name=key, mode="lines", line=dict(color=color, width=2),
                        fill="tozeroy", fillcolor=hex_to_rgba(color, 0.12),
                        hovertemplate=key + ": %{x:,.0f} €<extra></extra>")
    top = 0
    for tr in fig.data:
        top = max(top, max(tr.y))
    start = stats_ziel["start"]
    fig.add_vline(x=start, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=start, y=top * 1.05, text=f"heute {start:,.0f} €", showarrow=False,
                       font=dict(color=MUTED, size=12), yanchor="bottom")
    for q, label in (("p5", "P5"), ("p50", "P50"), ("p95", "P95")):
        fig.add_vline(x=stats_ziel[q], line=dict(color=ZIEL, width=1, dash="dash"))
        fig.add_annotation(x=stats_ziel[q], y=top * 0.92, text=f"{label} {stats_ziel[q]:,.0f} €", showarrow=False,
                           font=dict(color=MUTED, size=11), xanchor="left", xshift=4)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0, title=None))
    fig.update_xaxes(title=f"Depotwert in {horizon} Jahren (€)", tickformat=",.0f")
    fig.update_yaxes(showticklabels=False, title=None, showgrid=False)
    return _layout(fig, 360)


def upside_chart(t: pd.DataFrame) -> go.Figure:
    """Modell-Potenzial (P50/Kurs − 1) je Titel, Pole blau (+) / orange (−); Mischung als Marker."""
    d = t[t["Firma"] != "Cash (EUR)"].sort_values("Upside P50")
    names = [n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "") for n in d["Firma"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=d["Upside P50"], orientation="h", name="Modell (P50)",
                marker=dict(color=[POS if v >= 0 else NEG for v in d["Upside P50"]], cornerradius=4),
                text=[f"{v:+.0%}" for v in d["Upside P50"]], textposition="outside", cliponaxis=False,
                textfont=dict(color=MUTED, size=12), hovertemplate="%{y}<br>Modell: %{x:+.0%}<extra></extra>")
    fig.add_scatter(y=names, x=d["E[R] Horizont"], mode="markers", name="Erwartet über den Horizont",
                    marker=dict(symbol="diamond", size=10, color="#1f2633", line=dict(color="white", width=2)),
                    hovertemplate="%{y}<br>Erwartet: %{x:+.0%}<extra></extra>")
    fig.add_vline(x=0, line=dict(color="#c3c9d3", width=1))
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=None), bargap=0.4)
    fig.update_xaxes(tickformat="+.0%", title=None)
    return _layout(fig, max(340, 30 * len(names) + 90))


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Korrelation der Wochenrenditen (EUR) als divergierende Heatmap um 0."""
    short = [n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "")[:18] for n in corr.index]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short, zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, NEG], [0.5, MID], [1, POS]], xgap=2, ygap=2,
        text=np.round(corr.values, 2), texttemplate="%{text:.2f}", textfont=dict(size=11),
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10, len=0.8)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _layout(fig, max(380, 28 * len(short) + 140))
