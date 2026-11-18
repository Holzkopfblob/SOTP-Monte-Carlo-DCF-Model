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


# Gruppenfarben: dataviz-Referenzpalette in fester Reihenfolge, je Gruppe fest zugeordnet (Farbe folgt der Gruppe,
# nicht dem Rang). Validiert (validate_palette.js, light): CVD worst 9,1, Normalsicht 19,6; Kontrast < 3:1 bei
# Aqua/Gelb/Magenta ⇒ Donut-Scheiben immer mit Namen und Prozent beschriftet. Cash = neutrales Grau.
GROUP_COLORS = {
    "Telko": "#2a78d6", "Halbleiter/Elektronik": "#eb6834", "Industrie": "#1baf7a", "Gesundheit": "#eda100",
    "Software/IT": "#e87ba4", "Auto": "#008300", "Konsum": "#6250d6",
}
EXTRA_COLORS = ["#e34948", "#8a6d3b", "#4b5563"]     # weitere Gruppen (selten); mehr ⇒ in „Sonstige“ falten
CASH_COLOR = "#a3abb8"
CASH = "Cash (EUR)"


def _short(n: str) -> str:
    return n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "") \
            .replace(" Group Plc", "").replace(" Holding N.V.", "").replace(" S.A.", "").replace(" SE", "")             .replace(" Software Technologies Ltd.", "").replace(" Communications", "").replace(" Group", "")


def group_weights(t: pd.DataFrame) -> pd.DataFrame:
    """Ist- und Ziel-Gewicht je Gruppe (Cash als eigene Zeile), Anzahl Titel > 0,5 % je Seite."""
    d = t.copy()
    d["Gruppe"] = np.where(d["Firma"] == CASH, CASH, d["Gruppe"].fillna("Ohne Gruppe"))
    g = d.groupby("Gruppe")[["Ist-Gewicht", "Ziel-Gewicht"]].sum()
    return g.loc[(g.abs() > 1e-4).any(axis=1)]


def _group_color(name: str, others: list[str]) -> str:
    if name == CASH:
        return CASH_COLOR
    if name in GROUP_COLORS:
        return GROUP_COLORS[name]
    return EXTRA_COLORS[others.index(name) % len(EXTRA_COLORS)]


def group_donuts(t: pd.DataFrame) -> go.Figure:
    """Zwei Donuts (Ist | Ziel) nach Gruppe; gleiche Gruppe = gleiche Farbe, Cash grau, 2px Lücke zwischen Scheiben.
    Mitte: Zahl der Titel über 0,5 %."""
    from plotly.subplots import make_subplots
    g = group_weights(t)
    order = [x for x in GROUP_COLORS if x in g.index] + sorted(x for x in g.index if x not in GROUP_COLORS and x != CASH)
    order += [CASH] if CASH in g.index else []
    others = [x for x in order if x not in GROUP_COLORS and x != CASH]
    colors = [_group_color(x, others) for x in order]
    titles = t[t["Firma"] != CASH]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], horizontal_spacing=0.06)
    for i, (col, label) in enumerate((("Ist-Gewicht", "Ist"), ("Ziel-Gewicht", "Ziel")), start=1):
        vals = [max(float(g.loc[x, col]), 0.0) for x in order]
        n = int((titles[col] > 0.005).sum())
        fig.add_pie(labels=order, values=vals, hole=0.62, sort=False, direction="clockwise", name=label,
                    marker=dict(colors=colors, line=dict(color="white", width=2)),
                    text=[f"{x}<br>{v / sum(vals):.0%}" if sum(vals) and v / sum(vals) >= 0.01 else "" for x, v in zip(order, vals)],
                    textinfo="text", textposition="outside", textfont=dict(color=MUTED, size=12),
                    hovertemplate=label + " · %{label}<br>%{percent:.1%}<extra></extra>", row=1, col=i)
        fig.add_annotation(text=f"<b>{label}</b><br>{n} Titel", x=0.22 if i == 1 else 0.78, y=0.5, showarrow=False,
                           font=dict(size=14, color="#1f2633"), xref="paper", yref="paper")
    fig.update_traces(showlegend=False)
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    _layout(fig, 380)
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def dumbbell_chart(t: pd.DataFrame) -> go.Figure:
    """Ist → Ziel je Titel: grauer Punkt = Ist, farbiger Punkt = Ziel (blau aufstocken, orange abbauen), Linie dazwischen.
    Sortiert nach Ziel; Cash unten; Titel ohne Ist und Ziel entfallen."""
    d = t[(t["Ist-Gewicht"].abs() >= 0.005) | (t["Ziel-Gewicht"].abs() >= 0.005)].copy()   # < 0,5 % beidseitig: weglassen
    cash = d[d["Firma"] == CASH]
    d = pd.concat([cash, d[d["Firma"] != CASH].sort_values(["Ziel-Gewicht", "Ist-Gewicht"])])
    names = [_short(n) for n in d["Firma"]]
    ist, ziel = d["Ist-Gewicht"].to_numpy(), d["Ziel-Gewicht"].to_numpy()
    up = ziel > ist + 1e-3
    down = ziel < ist - 1e-3
    fig = go.Figure()
    for mask, color in ((up, POS), (down, NEG)):
        xs, ys = [], []
        for n, a_, b_ in zip(np.array(names)[mask], ist[mask], ziel[mask]):
            xs += [a_, b_, None]
            ys += [n, n, None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=hex_to_rgba(color, 0.45), width=2),
                        hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ist, y=names, mode="markers", name="Ist",
                    marker=dict(size=10, color="#a3abb8", line=dict(color="white", width=2)),
                    hovertemplate="%{y}<br>Ist: %{x:.1%}<extra></extra>")
    zc = np.where(up, POS, np.where(down, NEG, "#5b6475"))
    fig.add_scatter(x=ziel, y=names, mode="markers+text", name="Ziel", showlegend=False,
                    marker=dict(size=12, color=zc, line=dict(color="white", width=2)),
                    text=[f"{z:.0%}" if abs(z - i) > 1e-3 or z > 0.004 else "" for z, i in zip(ziel, ist)],
                    textposition="middle right", textfont=dict(color=MUTED, size=12), cliponaxis=False,
                    customdata=np.column_stack([ist, ziel - ist]),
                    hovertemplate="%{y}<br>Ziel: %{x:.1%} (Ist %{customdata[0]:.1%}, Δ %{customdata[1]:+.1%})<extra></extra>")
    # Legendeneinträge für die Richtung (Farbe nie allein: Legende + Prozent-Beschriftung)
    for color, label in ((POS, "Ziel · aufstocken"), (NEG, "Ziel · abbauen")):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=label, marker=dict(size=12, color=color))
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0, title=None))
    hi = float(max(ist.max(initial=0), ziel.max(initial=0)))
    fig.update_xaxes(tickformat=".0%", title=None, range=[-0.012, hi * 1.12 + 0.01])   # Luft links: Punkte bei 0 % frei
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=names, showgrid=True, ticksuffix="  ")
    return _layout(fig, max(320, 30 * len(names) + 90))


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


def holdings_chart(held: pd.DataFrame) -> go.Figure:
    """Bestand: Gewinn/Verlust je Position in € (Balken, blau/orange), Text = in % vom Einstand; ohne Einstand grau."""
    d = held.sort_values("Wert €")
    names = [n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "") for n in d["Firma"]]
    g = d["Gewinn €"].fillna(0.0)
    fig = go.Figure()
    fig.add_bar(y=names, x=g, orientation="h",
                marker=dict(color=["#c3c9d3" if np.isnan(v) else POS if v >= 0 else NEG for v in d["Gewinn €"]],
                            cornerradius=4),
                text=["Einstand fehlt" if np.isnan(v) else f"{v:+,.0f} € ({pc:+.1%})".replace(",", ".")
                      for v, pc in zip(d["Gewinn €"], d["Gewinn %"])],
                textposition="outside", cliponaxis=False, textfont=dict(color=MUTED, size=12),
                customdata=np.column_stack([d["Wert €"], d["Gewicht"]]),
                hovertemplate="%{y}<br>Gewinn: %{x:,.0f} €<br>Wert %{customdata[0]:,.0f} € · %{customdata[1]:.1%}"
                              "<extra></extra>")
    fig.add_vline(x=0, line=dict(color="#c3c9d3", width=1))
    fig.update_layout(showlegend=False, bargap=0.4)
    fig.update_xaxes(tickformat=",.0f", title="Gewinn/Verlust seit Kauf (€)")
    return _layout(fig, max(260, 34 * len(names) + 90))


def return_chart(t: pd.DataFrame) -> go.Figure:
    """Faktor-Modus: erwartete Rendite p. a. (Median) je Titel; Raute = eigene Hürde rf + β·ERP (CAPM).
    Blau = Rendite über der Hürde (positives Alpha, unterbewertet), orange = darunter; Text = Alpha in pp."""
    d = t[t["Firma"] != "Cash (EUR)"].sort_values("Vorsprung p. a.")
    names = [n.replace(", Inc.", "").replace(" Inc.", "").replace(" AG", "").replace(" Corporation", "") for n in d["Firma"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=d["Rendite p. a."], orientation="h", name="Erwartete Rendite p. a. (Median)",
                marker=dict(color=[POS if v > 0 else NEG for v in d["Vorsprung p. a."]], cornerradius=4),
                text=[f"α {e * 100:+.1f} pp" for e in d["Vorsprung p. a."]],
                textposition="outside", cliponaxis=False, textfont=dict(color=MUTED, size=12),
                customdata=np.column_stack([d["Ziel-Gewicht"], d["Beta"]]),
                hovertemplate="%{y}<br>Rendite p. a.: %{x:.1%}<br>β %{customdata[1]:.2f} · Ziel %{customdata[0]:.1%}"
                              "<extra></extra>")
    fig.add_scatter(y=names, x=d["Hürde p. a."], mode="markers", name="Hürde rf + β·ERP",
                    marker=dict(symbol="diamond", size=10, color="#1f2633", line=dict(color="white", width=2)),
                    hovertemplate="%{y}<br>Hürde: %{x:.1%}<extra></extra>")
    fig.add_vline(x=0, line=dict(color="#c3c9d3", width=1))
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=None), bargap=0.4)
    fig.update_xaxes(tickformat=".0%", title=None)
    return _layout(fig, max(340, 30 * len(names) + 90))


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
