"""Diagramme der Kalibrierungs-Ansicht (Farben wie Portfolio: Modell = primary, Ist = secondary)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from presentation.charts.common import COLORS, hex_to_rgba
from presentation.charts.portfolio import MUTED, _layout

MODEL, IST = COLORS["primary"], COLORS["secondary"]


def interval_chart(ev: pd.DataFrame) -> go.Figure:
    """Je Prognose: Modell-Band P5–P95 (Balken) mit P50 (Strich) und Ist-Wert (Punkt), sortiert nach Rang."""
    d = ev.sort_values("pit").reset_index(drop=True)
    labels = [f"{f} · {x}" for f, x in zip(d["firma"], d["feld"])]
    fig = go.Figure()
    fig.add_bar(y=labels, x=d["p95"] - d["p5"], base=d["p5"], orientation="h", name="Modell P5–P95",
                marker=dict(color=hex_to_rgba(MODEL, 0.28), line=dict(color=MODEL, width=1), cornerradius=4),
                customdata=np.c_[d["p5"], d["p50"], d["p95"]],
                hovertemplate="%{y}<br>Modell P5 %{customdata[0]:.1f} % · P50 %{customdata[1]:.1f} % · "
                              "P95 %{customdata[2]:.1f} %<extra></extra>")
    fig.add_scatter(y=labels, x=d["p50"], mode="markers", name="Modell P50",
                    marker=dict(symbol="line-ns", size=14, line=dict(color=MODEL, width=3)), hoverinfo="skip")
    fig.add_scatter(y=labels, x=d["ist_pct"].astype(float), mode="markers+text", name="Ist",
                    marker=dict(size=11, color=IST, line=dict(color="white", width=2)),
                    text=[f"{v:+.1f} %" for v in d["ist_pct"].astype(float)], textposition="middle right",
                    textfont=dict(color=MUTED, size=11),
                    customdata=d["pit"], hovertemplate="%{y}<br>Ist %{x:.1f} % · Rang %{customdata:.2f}<extra></extra>")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=None), bargap=0.45)
    fig.update_xaxes(title="Umsatzwachstum Jahr 1 (%)", ticksuffix=" %")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, title=None)
    return _layout(fig, max(320, 34 * len(labels) + 90))


def pit_histogram(pits: list[float]) -> go.Figure:
    """Verteilung der Quantil-Ränge in Dezilen; gestrichelt die Erwartung bei guter Kalibrierung."""
    counts, edges = np.histogram(pits, bins=10, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, width=0.085, marker=dict(color=MODEL, cornerradius=4),
                           text=[str(c) if c else "" for c in counts], textposition="outside", cliponaxis=False,
                           textfont=dict(color=MUTED), hovertemplate="Rang %{x:.2f} ± 0,05: %{y}<extra></extra>",
                           showlegend=False))
    expect = len(pits) / 10
    fig.add_hline(y=expect, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x=1.0, y=expect, text=f"ideal {expect:.1f} je Dezil", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(color=MUTED, size=11))
    fig.update_xaxes(title="Quantil-Rang des Ist-Werts in der Modellverteilung", range=[0, 1], dtick=0.1)
    fig.update_yaxes(title="Anzahl", rangemode="tozero")
    return _layout(fig, 300)
