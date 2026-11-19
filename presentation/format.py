"""German number formatting for all user-facing text (decimal comma, thousands dot).

Charts get the same format via the Plotly template (presentation.charts.common), tables via
st.column_config formats "localized"/"euro"/"percent" (browser locale).
"""
from __future__ import annotations

import math


def num(x: float | None, dec: int = 0, signed: bool = False) -> str:
    """1234.5 → '1.234,5'; None/NaN → '–'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:{'+' if signed else ''},.{dec}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def pct(x: float | None, dec: int = 1, signed: bool = False) -> str:
    """0.123 → '12,3 %'."""
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{num(x * 100, dec, signed)} %"


def eur(x: float | None, dec: int = 0) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{num(x, dec)} €"


def mio(x: float) -> str:
    """Amounts in Mio.; from 10.000 Mio. in Mrd. so metric cards don't truncate."""
    return f"{num(x / 1_000, 1)} Mrd." if abs(x) >= 10_000 else f"{num(x)} Mio."
