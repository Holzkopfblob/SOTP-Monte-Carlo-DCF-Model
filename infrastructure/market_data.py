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


BUND_3Y_URL = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
               "D.I.ZST.ZI.EUR.S1311.B.A604.R03XX.R.A.A._Z._Z.A?format=csv&lang=de&lastNObservations=1")


def fetch_bund_3y() -> tuple[float, str]:
    """Rendite Bund 3 J. (Bundesbank-Zinsstruktur, Svensson) als Dezimalzahl und Datum."""
    from urllib.request import urlopen

    with urlopen(BUND_3Y_URL, timeout=15) as r:
        last = r.read().decode("utf-8-sig").strip().splitlines()[-1]
    day, value = last.split(";")[:2]
    return float(value.replace(",", ".")) / 100, day
