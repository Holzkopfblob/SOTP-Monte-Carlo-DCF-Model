"""Allocation charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .common import COLORS, apply_figure_defaults


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
    """SOTP value-bridge waterfall."""
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

    if abs(minority_interests) > 0.01:
        names.append("Minderheitsanteile")
        values.append(-minority_interests)
        measures.append("relative")
    if abs(pension_liabilities) > 0.01:
        names.append("Pensionsrückstellungen")
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

    apply_figure_defaults(
        fig,
        title="SOTP-Wertbrücke (Erwartungswerte / Mio.)",
        yaxis_title="Wert (Mio.)",
        height=520,
    )
    fig.update_layout(showlegend=False)
    return fig


