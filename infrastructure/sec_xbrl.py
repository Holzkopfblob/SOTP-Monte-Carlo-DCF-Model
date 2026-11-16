"""
SEC-XBRL-Datenschicht (v3-Plan, Phase 2): Jahreswerte aller US-GAAP-Einreicher über die Frames-API
(ein Abruf liefert eine Kennzahl für ein Kalenderjahr über alle Firmen) und die SIC-Branche je Firma.

Cache außerhalb des Repos: Umgebungsvariable SOTP_DATA (Standard 05_PRIVAT/Finanzen/sotp-daten).
SEC-Regeln: höchstens 10 Abrufe/s, User-Agent mit Kontakt; wir senden einen generischen Agenten ohne
persönliche Daten (wie bei den bisherigen SEC-Abrufen).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

UA = "SOTP-Research-Tool contact@sotp-research.invalid"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/CY{year}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
PAUSE = 0.12   # < 10 Abrufe/s

# Kennzahl → XBRL-Tags in Vorrangreihenfolge (Firmen melden unterschiedlich; vor/nach ASC 606)
CONCEPTS = {
    "umsatz": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
               "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "ebit": ["OperatingIncomeLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
           "DepreciationAmortizationAndAccretionNet"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def data_dir() -> Path:
    d = Path(os.environ.get("SOTP_DATA", r"C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen\sotp-daten"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:          # Tag in diesem Jahr nicht vorhanden
            return None
        raise
    finally:
        time.sleep(PAUSE)


def fetch_frame(tag: str, year: int) -> pd.DataFrame:
    """Ein Frame (Tag × Kalenderjahr), gecacht als Parquet. Spalten: cik, name, loc, start, end, val."""
    f = data_dir() / "frames" / f"{tag}_CY{year}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = _get_json(FRAMES_URL.format(tag=tag, year=year))
    df = pd.DataFrame(d["data"] if d else [], columns=["cik", "entityName", "loc", "start", "end", "val"])
    df = df.rename(columns={"entityName": "name"})
    f.parent.mkdir(exist_ok=True)
    df.to_parquet(f)
    return df


def build_panel(years: range = range(2009, 2026)) -> pd.DataFrame:
    """Jahres-Panel cik × Jahr mit umsatz, ebit, da, capex (USD). Je Kennzahl gewinnt der erste Tag mit Wert."""
    parts = []
    for concept, tags in CONCEPTS.items():
        for year in years:
            got = None
            for tag in tags:
                df = fetch_frame(tag, year)[["cik", "name", "loc", "val"]]
                df = df[~df.cik.isin(got.cik)] if got is not None else df
                got = df if got is None else (pd.concat([got, df]) if len(df) else got)
            parts.append(got.assign(jahr=year, kennzahl=concept))
    long = pd.concat(parts, ignore_index=True)
    panel = long.pivot_table(index=["cik", "jahr"], columns="kennzahl", values="val", aggfunc="first").reset_index()
    names = long.drop_duplicates("cik", keep="last").set_index("cik")[["name", "loc"]]
    return panel.join(names, on="cik")


def fetch_sic(ciks) -> pd.Series:
    """SIC-Code je CIK (gecacht in sic.parquet; nur fehlende werden abgerufen)."""
    f = data_dir() / "sic.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["cik", "sic"])
    todo = sorted(set(int(c) for c in ciks) - set(have.cik.astype(int)))
    new = []
    for i, cik in enumerate(todo):
        d = _get_json(SUBMISSIONS_URL.format(cik=cik)) or {}
        new.append({"cik": cik, "sic": str(d.get("sic") or "")})
        if (i + 1) % 250 == 0:      # Zwischenstand sichern (Abbruch-fest)
            pd.concat([have, pd.DataFrame(new)]).to_parquet(f)
    out = pd.concat([have, pd.DataFrame(new)], ignore_index=True) if new else have
    out.to_parquet(f)
    return out.set_index("cik").sic


def load_panel(min_years: int = 6) -> pd.DataFrame:
    """Panel mit SIC (aus dem Cache bauen bzw. ergänzen); nur Firmen mit ≥ min_years Umsatzjahren."""
    f = data_dir() / "panel.parquet"
    panel = pd.read_parquet(f) if f.exists() else build_panel()
    if not f.exists():
        panel.to_parquet(f)
    n_years = panel.dropna(subset=["umsatz"]).groupby("cik").jahr.nunique()
    panel = panel[panel.cik.isin(n_years[n_years >= min_years].index)]
    sic = fetch_sic(panel.cik.unique())
    return panel.assign(sic=panel.cik.map(sic))
