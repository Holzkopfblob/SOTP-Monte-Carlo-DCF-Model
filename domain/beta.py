"""
Beta je Titel und Jensens Alpha für die Depot-Hürde (18.11.2026).

1. Regression (OLS) der Wochen-Log-Renditen des Titels in EUR auf den Weltmarkt in EUR (MSCI World, EUNL.DE):
   Beta, Standardfehler, Restrisiko (Varianz der Residuen, annualisiert).
2. Bayes-Schrumpfung nach Vasicek (1973): β_post = w·β_OLS + (1 − w)·β_prior, w = σ²_prior / (σ²_prior + SE²).
   Unsichere Regressionen werden stärker zum Branchen-Prior gezogen (schlägt die pauschale Blume-Anpassung).
3. Prior = Damodaran-Branchenbeta (Januar 2026, Region nach Währung), über Harris–Pringle auf den Hebel des Titels
   umgerechnet: β_A = E/V·β_E + D/V·β_D (konstante Schuldenquote, konsistent mit der WACC-Formel), β_E = (β_A − D/V·β_D)/(E/V).
4. Alpha = erwartete Rendite − (rf + β·ERP); Anker nach Treynor–Black (1973): a_i ∝ α_i⁺ / σ²_ε,i.

Reine Funktionen ohne I/O bis auf das Laden der Damodaran-Tabelle (data/damodaran_betas_2026-01.csv).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DAMODARAN_CSV = Path(__file__).resolve().parents[1] / "data" / "damodaran_betas_2026-01.csv"
MARKET_TICKER = "EUNL.DE"          # iShares Core MSCI World, EUR: Marktportfolio eines EUR-Anlegers mit Weltdepot
PRIOR_SD = 0.30                    # Querschnitts-Streuung der Betas innerhalb einer Branche
# ponytail: PRIOR_SD ist eine Literatur-Größenordnung (0,3–0,5), nicht aus eigenen Daten geschätzt; Upgrade:
# Streuung der Regressions-Betas je Branche aus dem SEC-Panel.

# Depot-Sektor → Damodaran-Branche (grob; die Regression dominiert bei 3 J. Wochendaten mit ~85–95 % Gewicht)
SEKTOR_BRANCHE = {
    "Telko": "Telecom. Services",
    "Software/IT": "Software (System & Application)",
    "Halbleiter/Elektronik": "Semiconductor",
    "Auto": "Auto & Truck",
    "Gesundheit": "Healthcare Support Services",
    "Industrie": "Machinery",
    "Immobilien": "Real Estate (Operations & Services)",
}
FALLBACK_BRANCHE = "Total Market (without financials)"


def debt_beta(d_v: float) -> float:
    """Debt-Beta nach Rating-Klasse (Berk/DeMarzo nach Schaefer/Strebulaev 2008: A 0,05, BBB 0,10, BB 0,17),
    Rating mangels Daten aus der Schuldenquote genähert. Nettokasse ⇒ 0."""
    # ponytail: Rating-Proxy über D/V; Upgrade: echtes Emittentenrating je Firma in firmen.csv
    if d_v <= 0:
        return 0.0
    return 0.05 if d_v < 0.2 else 0.10 if d_v < 0.5 else 0.17


def unlever(beta_e: float, d_v: float, beta_d: float) -> float:
    """Harris–Pringle: Asset-Beta aus Equity-Beta bei konstanter Schuldenquote D/V."""
    d_v = max(d_v, 0.0)
    return (1 - d_v) * beta_e + d_v * beta_d


def relever(beta_a: float, d_v: float, beta_d: float) -> float:
    """Harris–Pringle umgekehrt: Equity-Beta bei Schuldenquote D/V."""
    d_v = min(max(d_v, 0.0), 0.95)
    return (beta_a - d_v * beta_d) / (1 - d_v)


@lru_cache(maxsize=1)
def _damodaran() -> pd.DataFrame:
    return pd.read_csv(DAMODARAN_CSV)


def industry_asset_beta(sektor: str | None, waehrung: str) -> tuple[float, str]:
    """Asset-Beta der Branche (Harris–Pringle aus Damodarans levered Beta und D/E), Region nach Währung."""
    region = "US" if waehrung == "USD" else "Europa"
    branche = SEKTOR_BRANCHE.get(sektor or "", FALLBACK_BRANCHE)
    t = _damodaran()
    row = t[(t.region == region) & (t.branche == branche)].iloc[0]
    d_v = row.d_e / (1 + row.d_e)
    return unlever(row.beta, d_v, debt_beta(d_v)), f"Damodaran {region} {branche} (Jan. 2026)"


def ols_beta(r_i: np.ndarray, r_m: np.ndarray, periods: int = 52) -> dict:
    """OLS r_i = a + β·r_m + ε auf gemeinsamen Beobachtungen. Restrisiko annualisiert (Varianz × periods)."""
    r_i, r_m = np.asarray(r_i, float), np.asarray(r_m, float)
    ok = np.isfinite(r_i) & np.isfinite(r_m)
    x, y = r_m[ok], r_i[ok]
    n = len(x)
    if n < 20:
        return {"beta": np.nan, "se": np.inf, "resid_var": np.nan, "n": n}
    xm = x - x.mean()
    b = float(xm @ (y - y.mean()) / (xm @ xm))
    resid = y - y.mean() - b * xm
    s2 = float(resid @ resid / (n - 2))
    return {"beta": b, "se": float(np.sqrt(s2 / (xm @ xm))), "resid_var": s2 * periods, "n": n}


def vasicek(beta_ols: float, se: float, prior: float, prior_sd: float = PRIOR_SD) -> tuple[float, float]:
    """Bayes-Schrumpfung (Vasicek 1973). Rückgabe: (β_post, Gewicht der Regression)."""
    if not np.isfinite(beta_ols) or not np.isfinite(se):
        return prior, 0.0
    w = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
    return w * beta_ols + (1 - w) * prior, w


def treynor_black_anchor(alpha: np.ndarray, resid_var: np.ndarray, budget: float, cash_floor: float = 0.02) -> np.ndarray:
    """Anker des robusten Kelly nach Treynor–Black: a_i ∝ α_i⁺ / σ²_ε,i, Cash als letzte Spalte; Summe 1."""
    a = np.clip(np.nan_to_num(np.asarray(alpha, float), nan=0.0), 0.0, None)
    v = np.nan_to_num(np.asarray(resid_var, float), nan=np.inf)
    s = np.where(a > 0, a / np.maximum(v, 1e-6), 0.0)
    c = max(1.0 - budget, cash_floor)
    w = (1.0 - c) * s / s.sum() if s.sum() > 0 else np.zeros_like(s)
    return np.append(w, 1.0 - w.sum())
