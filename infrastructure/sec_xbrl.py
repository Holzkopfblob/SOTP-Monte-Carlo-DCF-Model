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
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{unit}/{period}.json"
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


def fetch_frame(tag: str, year: int, *, period: str = "", tax: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """Ein Frame (Tag × Periode), gecacht als Parquet. Spalten: cik, name, loc, start, end, val.
    *period*: Suffix nach CY{year}, z. B. "Q4I" (Stichtag Jahresende) oder "Q2I"; leer = Kalenderjahr (Zeitraum)."""
    per = f"CY{year}{period}"
    pre = "" if (tax, unit) == ("us-gaap", "USD") else f"{tax}_{unit}_"
    f = data_dir() / "frames" / f"{pre}{tag}_{per}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = _get_json(FRAMES_URL.format(tax=tax, tag=tag, unit=unit, period=per))
    df = pd.DataFrame(d["data"] if d else [], columns=["cik", "entityName", "loc", "start", "end", "val"])
    df = df.rename(columns={"entityName": "name"})
    f.parent.mkdir(exist_ok=True)
    df.to_parquet(f)
    return df


# Rückblick-Labor (Baustein 6): Bilanz zum Jahresende (Q4I), Ausschüttungen/Emissionen (Jahr), Public Float (Q2I, dei)
LAB_CONCEPTS = {   # Kennzahl → (Tags in Vorrang, Periode, Taxonomie)
    "schulden_gesamt": (["DebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
                         "DebtLongtermAndShorttermCombinedAmount", "LongTermDebt"], "Q4I", "us-gaap"),
    "schulden_lang": (["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesPayable"],
                      "Q4I", "us-gaap"),
    "schulden_kurz": (["DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtCurrent"],
                      "Q4I", "us-gaap"),
    "kredite_kurz": (["ShortTermBorrowings", "CommercialPaper"], "Q4I", "us-gaap"),
    "kasse": (["CashAndCashEquivalentsAtCarryingValue",
               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "Q4I", "us-gaap"),
    "dividenden": (["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "", "us-gaap"),
    "rueckkauf": (["PaymentsForRepurchaseOfCommonStock"], "", "us-gaap"),
    "emission": (["ProceedsFromIssuanceOfCommonStock"], "", "us-gaap"),
    "steuern": (["IncomeTaxesPaidNet", "IncomeTaxesPaid"], "", "us-gaap"),
    "capex_alt": (["PaymentsToAcquireProductiveAssets", "PaymentsToAcquireOtherPropertyPlantAndEquipment",
                   "PaymentsForCapitalImprovements"], "", "us-gaap"),
    # Public Float gilt zum Ende des 2. Geschäftsquartals ⇒ je nach GJ-Ende ein anderes Kalenderquartal
    **{f"public_float_q{q}": (["EntityPublicFloat"], f"Q{q}I", "dei") for q in (1, 2, 3, 4)},
}


def build_lab_panel(years: range = range(2009, 2026)) -> pd.DataFrame:
    """Labor-Panel cik × Jahr mit den LAB_CONCEPTS (je Kennzahl gewinnt der erste Tag mit Wert); Nettoschulden =
    LongTermDebt (sonst lang + kurzfristiger Anteil) + kurzfristige Kredite − Kasse. Gecacht als labor_panel.parquet."""
    f = data_dir() / "labor_panel.parquet"
    if f.exists():
        return pd.read_parquet(f)
    parts = []
    for concept, (tags, period, tax) in LAB_CONCEPTS.items():
        for year in years:
            got = None
            for tag in tags:
                df = fetch_frame(tag, year, period=period, tax=tax)[["cik", "val"]]
                df = df[~df.cik.isin(got.cik)] if got is not None else df
                got = df if got is None else (pd.concat([got, df]) if len(df) else got)
            parts.append(got.assign(jahr=year, kennzahl=concept))
    long = pd.concat(parts, ignore_index=True)
    p = long.pivot_table(index=["cik", "jahr"], columns="kennzahl", values="val", aggfunc="first").reset_index()
    # ponytail: Schuldendefinition grob (ohne Leasing/Pensionen); Upgrade: OperatingLeaseLiability, Pensionsdefizit
    # Gesamttag (inkl. kurzfristiger Teile) zuerst, sonst lang + kurz; kurzfristige Kredite nur, wenn kein Kurz-Tag
    # (DebtCurrent enthält sie meist). Ohne jeden Schuldentag: NaN statt 0 (Fall fällt im Labor raus, nicht schuldenfrei).
    teil = p["schulden_lang"].fillna(0) + p["schulden_kurz"].fillna(p["kredite_kurz"]).fillna(0)
    # nur Kurz-Tag ⇒ Teilwert (z. B. Oracle, GJ-Ende Mai: Q4I-Frame greift die 10-Q-Bilanz ohne Langfristschulden)
    hat = p[["schulden_gesamt", "schulden_lang"]].notna().any(axis=1)
    p["schulden"] = p["schulden_gesamt"].fillna(teil).where(hat)
    p["nettoschulden"] = p["schulden"] - p["kasse"].fillna(0)
    # Börsenwert zum Stichtag 30.06. des Jahres: letzter Public Float im Fenster 01.07. Vorjahr … 30.06. (keine
    # Vorausschau; bis zu 12 Monate alt). ponytail: veralteter Kurs statt Tageskurs; Upgrade: Kursdaten je Firma.
    vj = p[["cik", "jahr", "public_float_q3", "public_float_q4"]].assign(jahr=p.jahr + 1)
    p = p.merge(vj, on=["cik", "jahr"], how="left", suffixes=("", "_vj"))
    p["public_float"] = (p["public_float_q2"].fillna(p["public_float_q1"]).fillna(p["public_float_q4_vj"])
                         .fillna(p["public_float_q3_vj"]))
    p.to_parquet(f)
    return p


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


def abgang_grund(ciks) -> pd.Series:
    """Warum eine Firma aufhörte zu melden (gecacht in abgang.parquet): 'insolvenz' (8-K Item 1.03 in den letzten
    24 Monaten der Einreichungen), 'uebernahme' (Item 2.01/5.01 in den letzten 12 Monaten), sonst 'unbekannt'."""
    f = data_dir() / "abgang.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["cik", "grund", "letzte"])
    todo = sorted(set(int(c) for c in ciks) - set(have.cik.astype(int)))
    new = []
    for i, cik in enumerate(todo):
        r = (_get_json(SUBMISSIONS_URL.format(cik=cik)) or {}).get("filings", {}).get("recent", {})
        df = pd.DataFrame({k: r.get(k, []) for k in ("form", "filingDate", "items")})
        grund, letzte = "unbekannt", ""
        if len(df):
            df["filingDate"] = pd.to_datetime(df.filingDate)
            end = df.filingDate.max()
            letzte = str(end.date())
            k8 = df[df.form.str.startswith("8-K")]
            if k8[k8.filingDate >= end - pd.DateOffset(months=24)]["items"].str.contains("1.03").any():
                grund = "insolvenz"
            elif k8[k8.filingDate >= end - pd.DateOffset(months=12)]["items"].str.contains("2.01|5.01").any():
                grund = "uebernahme"
        new.append({"cik": cik, "grund": grund, "letzte": letzte})
        if (i + 1) % 200 == 0:
            pd.concat([have, pd.DataFrame(new)]).to_parquet(f)
    out = pd.concat([have, pd.DataFrame(new)], ignore_index=True) if new else have
    out.to_parquet(f)
    return out.set_index("cik").grund


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
