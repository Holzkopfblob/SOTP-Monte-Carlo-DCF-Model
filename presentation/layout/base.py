"""Base layout helpers shared by Streamlit entry points."""

from __future__ import annotations

import streamlit as st

from presentation.theme.tokens import (
    APP_LAYOUT,
    APP_SIDEBAR_STATE,
    CARD_BORDER_RADIUS_PX,
    EXPANDER_BORDER_COLOR,
    EXPANDER_MARGIN_BOTTOM_PX,
    METRIC_BACKGROUND_COLOR,
    METRIC_PADDING,
    TAB_FONT_WEIGHT,
    TAB_GAP_PX,
    TAB_PADDING,
)


def configure_page(
    *,
    page_title: str,
    page_icon: str,
    layout: str = APP_LAYOUT,
    initial_sidebar_state: str = APP_SIDEBAR_STATE,
) -> None:
    """Configure Streamlit page settings for an app entry point."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )


def inject_global_styles(*, metric_border_color: str) -> None:
    """Inject shared global CSS with app-specific metric accent color."""
    st.markdown(
        f"""
<style>
    div[data-testid="stMetric"] {{
        background-color: {METRIC_BACKGROUND_COLOR};
        padding: {METRIC_PADDING};
        border-radius: {CARD_BORDER_RADIUS_PX}px;
        border-left: 4px solid {metric_border_color};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: {TAB_GAP_PX}px; }}
    .stTabs [data-baseweb="tab"] {{ padding: {TAB_PADDING}; font-weight: {TAB_FONT_WEIGHT}; }}
    details summary {{ font-weight: 600; }}
    div[data-testid="stExpander"] {{
        border: 1px solid {EXPANDER_BORDER_COLOR};
        border-radius: {CARD_BORDER_RADIUS_PX}px;
        margin-bottom: {EXPANDER_MARGIN_BOTTOM_PX}px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def render_app_header(*, title: str, caption: str) -> None:
    """Render the application title and subtitle/caption."""
    st.title(title)
    st.caption(caption)


def render_sidebar_footer(*, tech_stack: str) -> None:
    """Render a consistent footer in sidebars."""
    st.markdown("---")
    st.caption(tech_stack)
