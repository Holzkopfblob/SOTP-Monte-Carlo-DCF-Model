"""Kurshistorie über yfinance (Schlusskurse, dividendenbereinigt)."""
from __future__ import annotations

import pandas as pd

FX_TICKER = "EURUSD=X"


def fetch_closes(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """Tägliche Schlusskurse der Ticker plus EURUSD, Spalten = Ticker."""
    import yfinance as yf  # erst hier: App startet auch ohne Internet/yfinance

    syms = sorted(set(tickers) | {FX_TICKER})
    df = yf.download(syms, period=f"{years}y", progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(syms[0])
    missing = [t for t in syms if t not in df or df[t].dropna().empty]
    if missing:
        raise ValueError(f"Keine Kurse für: {', '.join(missing)}")
    return df
