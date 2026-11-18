"""Kurshistorie über yfinance (Schlusskurse, dividendenbereinigt)."""
from __future__ import annotations

import pandas as pd

FX_TICKER = "EURUSD=X"


def fetch_closes(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """Tägliche Schlusskurse der Ticker plus EURUSD und Weltmarkt (MSCI World in EUR, für Beta), Spalten = Ticker."""
    import yfinance as yf  # erst hier: App startet auch ohne Internet/yfinance

    from domain.beta import MARKET_TICKER
    syms = sorted(set(tickers) | {FX_TICKER, MARKET_TICKER})
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


TREASURY_3Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3&cosd={start}"


def fetch_treasury_3y() -> tuple[float, str]:
    """Rendite US-Treasury 3 J. (FRED DGS3, Prozent; Feiertage leer) als Dezimalzahl und Datum."""
    from datetime import date, timedelta
    from urllib.request import urlopen

    with urlopen(TREASURY_3Y_URL.format(start=date.today() - timedelta(days=30)), timeout=15) as r:
        rows = [ln.split(",") for ln in r.read().decode("utf-8-sig").strip().splitlines()[1:]]
    day, value = [x for x in rows if x[1].strip()][-1]
    return float(value) / 100, day
