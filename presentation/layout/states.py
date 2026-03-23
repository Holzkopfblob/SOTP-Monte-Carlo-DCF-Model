"""Standardized UI state renderers for consistent messaging."""

from __future__ import annotations

import streamlit as st


def render_empty_state(message: str, *, title: str | None = None) -> None:
    """Render an empty state with optional title."""
    if title:
        st.subheader(title)
    st.info(message)


def render_warning_state(message: str, *, title: str | None = None) -> None:
    """Render a warning state with optional title."""
    if title:
        st.subheader(title)
    st.warning(message)


def render_success_state(message: str, *, title: str | None = None) -> None:
    """Render a success state with optional title."""
    if title:
        st.subheader(title)
    st.success(message)
