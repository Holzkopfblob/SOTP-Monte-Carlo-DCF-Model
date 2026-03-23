"""Reusable insight card renderers for summary, risk and drivers."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


MetricTriplet = tuple[str, str, str | None]


def _render_metric_row(items: Sequence[MetricTriplet]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def render_summary_cards(items: Sequence[MetricTriplet]) -> None:
    """Render a summary metric card row."""
    if not items:
        return
    _render_metric_row(items)


def render_risk_cards(items: Sequence[MetricTriplet]) -> None:
    """Render a risk metric card row."""
    if not items:
        return
    _render_metric_row(items)


def render_driver_cards(items: Sequence[MetricTriplet]) -> None:
    """Render a driver metric card row."""
    if not items:
        return
    _render_metric_row(items)
