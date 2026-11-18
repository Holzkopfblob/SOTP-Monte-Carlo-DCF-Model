"""Kurshistorie über yfinance (Schlusskurse, dividendenbereinigt)."""
from __future__ import annotations

import pandas as pd

FX_TICKER = "EURUSD=X"


def fetch_closes(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """Tägliche Schlusskurse der Ticker plus EURUSD, Weltmarkt (MSCI World in EUR, für Beta) und Bund-ETFs (Zinsfaktor)."""
    import yfinance as yf  # erst hier: App startet auch ohne Internet/yfinance

    from domain.beta import MARKET_TICKER, TERM_TICKERS
    syms = sorted(set(tickers) | {FX_TICKER, MARKET_TICKER, *TERM_TICKERS})
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


DAMODARAN_HOME = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"
ERP_RUECKFALL = (0.0423, "2026-01-05")   # Damodaran Jan. 2026 [primär geprüft], falls nie ein Abruf gelang


def parse_damodaran_erp(html: str) -> tuple[float, str]:
    """„Implied ERP on November 1, 2026 = 4.41% (Trailing 12 month, with adjusted payout)…“ ⇒ (0,0441, "2026-11-01")."""
    import re
    from datetime import datetime

    text = re.sub(r"<[^>]+>", " ", html)
    m = re.search(r"Implied ERP on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*=\s*([\d.]+)\s*%", text)
    if not m:
        raise ValueError("Implied ERP nicht gefunden (Seitenaufbau geändert?)")
    day = datetime.strptime(re.sub(r"\s+", " ", m.group(1)), "%B %d, %Y").date().isoformat()
    return float(m.group(2)) / 100, day


def fetch_damodaran_erp() -> tuple[float, str, bool]:
    """Aktuelle implizite ERP (Damodaran, monatlich, S&P 500, Trailing 12 Monate mit angepasster Ausschüttung) als
    (Dezimalzahl, Datum, frisch). Erfolgreiche Abrufe landen in <SOTP_DATA>/erp.json; fällt der Abruf aus, gilt der
    letzte gespeicherte Wert (frisch = False)."""
    import json
    from urllib.request import Request, urlopen

    from infrastructure.sec_xbrl import UA, data_dir
    f = data_dir() / "erp.json"
    try:
        with urlopen(Request(DAMODARAN_HOME, headers={"User-Agent": UA}), timeout=15) as r:
            erp, day = parse_damodaran_erp(r.read().decode("utf-8", errors="replace"))
        f.write_text(json.dumps({"erp": erp, "datum": day}), encoding="utf-8")
        return erp, day, True
    except Exception:  # Netz oder Seitenaufbau: letzter bekannter Wert statt Abbruch
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return d["erp"], d["datum"], False
        return *ERP_RUECKFALL, False
