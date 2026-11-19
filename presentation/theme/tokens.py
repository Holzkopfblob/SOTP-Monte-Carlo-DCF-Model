"""Centralized style and color tokens for Streamlit presentation."""

from __future__ import annotations

APP_LAYOUT = "wide"
APP_SIDEBAR_STATE = "expanded"

METRIC_BACKGROUND_COLOR = "#f3f5f9"
METRIC_PADDING = "12px 16px"
CARD_BORDER_RADIUS_PX = 8

METRIC_BORDER_COLOR_DCF = "#2457c5"

TAB_GAP_PX = 4
TAB_PADDING = "8px 20px"
TAB_FONT_WEIGHT = 500

EXPANDER_BORDER_COLOR = "#e3e7ee"
EXPANDER_MARGIN_BOTTOM_PX = 8

# German number format in every chart (decimal comma, thousands dot) plus the app font.
# "plotly_white+sotp" layers our tweaks over plotly_white; registered in presentation.charts.common.
PLOTLY_TEMPLATE = "plotly_white+sotp"
PLOTLY_SEPARATORS = ",."
PLOTLY_FONT = dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color="#1b2333")

CHART_COLORS = {
    "primary": "#2457c5",
    "secondary": "#e8853d",
    "positive": "#2f9e6e",
    "negative": "#d1495b",
    "neutral": "#8792a2",
    "accent": "#7a5cc7",
    "tertiary": "#2a9fb5",
    "quaternary": "#c95f9e",
    "danger_bg": "#fee0d2",
    "warning_bg": "#fff3cd",
    "success_bg": "#d4edda",
}
