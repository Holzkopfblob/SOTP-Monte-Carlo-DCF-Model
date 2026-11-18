"""
Beta je Titel und Jensens Alpha für die Depot-Hürde (18.11.2026).

1. Regression (OLS) der Wochen-Log-Renditen des Titels in EUR auf den Weltmarkt in EUR (MSCI World, EUNL.DE):
   Beta, Standardfehler, Restrisiko (Varianz der Residuen, annualisiert).
2. Bayes-Schrumpfung nach Vasicek (1973): β_post = w·β_OLS + (1 − w)·β_prior, w = σ²_prior / (σ²_prior + SE²).
   Unsichere Regressionen werden stärker zum Branchen-Prior gezogen (schlägt die pauschale Blume-Anpassung).
3. Prior = Damodaran-Branchenbeta (Januar 2026, Region nach Währung), über Harris–Pringle auf den Hebel des Titels
   umgerechnet: β_A = E/V·β_E + D/V·β_D (konstante Schuldenquote, konsistent mit der WACC-Formel), β_E = (β_A − D/V·β_D)/(E/V).
4. Alpha = erwartete Rendite − (rf + β·ERP + β_z·TP); Anker nach Treynor–Black (1973): a_i ∝ α_i⁺ / σ²_ε,i.
5. Zinsfaktor (18.11.2026, Messung Documents/KALIBRIERUNG-2026-11/zinsfaktor_messung.py): TERM = Bund 5,5–10,5 J.
   minus Bund 1,5–2,5 J. (Fama-French). Zwei-Faktor-OLS, β_z per empirischem Bayes zur 0 geschrumpft (Streuung der
   wahren β_z = Querschnittsvarianz der Schätzer − mittlere SE²). Dieselbe Formel (equity_beta) gilt in den Bewertungen.

Reine Funktionen ohne I/O bis auf das Laden der Damodaran-Tabelle (data/damodaran_betas_2026-01.csv).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DAMODARAN_CSV = Path(__file__).resolve().parents[1] / "data" / "damodaran_betas_2026-01.csv"
MARKET_TICKER = "EUNL.DE"          # iShares Core MSCI World, EUR: Marktportfolio eines EUR-Anlegers mit Weltdepot
TERM_TICKERS = ("EXHD.DE", "EXHB.DE")   # iShares eb.rexx Germany 5,5–10,5 J. und 1,5–2,5 J. (EUR, thesaurierend)
# Erwartete TERM-Rendite p. a. Spanne: 0,15 % (Steigung Bund 7 J. − 2 J., Bundesbank 18.11.2026: 3,27 − 3,12) bis
# ~1 %; 0,5 % ≈ Mitte (Laufzeitprämie 10 J. ~1,1 % laut Oxford Economics 2026 [sekundär], anteilig 7 J. − 2 J.).
# ponytail: kein primär belegtes Niveau (EZB-Anhänge zeigen nur Veränderungen); Upgrade: eigene ATSM-Schätzung.
TERM_PREMIUM = 0.005
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


def industry_asset_beta(sektor: str | None, waehrung: str, branche: str | None = None) -> tuple[float, str]:
    """Asset-Beta der Branche (Harris–Pringle aus Damodarans levered Beta und D/E), Region nach Währung.
    *branche*: Damodaran-Branche der Firma (metadata.beta_branche); sonst grob über den Sektor."""
    region = "US" if waehrung == "USD" else "Europa"
    branche = branche or SEKTOR_BRANCHE.get(sektor or "", FALLBACK_BRANCHE)
    t = _damodaran()
    row = t[(t.region == region) & (t.branche == branche)].iloc[0]
    d_v = row.d_e / (1 + row.d_e)
    return unlever(row.beta, d_v, debt_beta(d_v)), f"Damodaran {region} {branche} (Jan. 2026)"


def equity_beta(sektor: str | None, waehrung: str, d_v: float, branche: str | None = None) -> tuple[float, str]:
    """Fundamentales Equity-Beta: Branchen-Asset-Beta über Harris–Pringle mit Schuldenbeta auf D/V umgerechnet.
    Eine Formel für Bewertung (WACC) und Portfolio (Prior der Hürde)."""
    b_a, src = industry_asset_beta(sektor, waehrung, branche)
    d_v = max(d_v, 0.0)
    return relever(b_a, d_v, debt_beta(d_v)), f"{src}, β_A {b_a:.2f}, D/V {d_v:.0%}, β_D {debt_beta(d_v):.2f}"


def term_factor(closes: pd.DataFrame) -> pd.Series | None:
    """Wochen-Log-Rendite TERM = lange minus kurze Bunds; None, wenn die Reihen fehlen."""
    if not all(t in closes for t in TERM_TICKERS):
        return None
    w = np.log(closes[list(TERM_TICKERS)].resample("W-FRI").last()).diff()
    return w[TERM_TICKERS[0]] - w[TERM_TICKERS[1]]


def ols_2f(r_i: np.ndarray, r_m: np.ndarray, r_z: np.ndarray, periods: int = 52) -> dict:
    """OLS r_i = a + β·r_m + β_z·r_z + ε. Standardfehler beider Steigungen, Restrisiko annualisiert."""
    r_i, r_m, r_z = (np.asarray(x, float) for x in (r_i, r_m, r_z))
    ok = np.isfinite(r_i) & np.isfinite(r_m) & np.isfinite(r_z)
    n = int(ok.sum())
    if n < 30:
        return {"beta": np.nan, "se": np.inf, "beta_z": np.nan, "se_z": np.inf, "resid_var": np.nan, "n": n}
    X = np.column_stack([np.ones(n), r_m[ok], r_z[ok]])
    y = r_i[ok]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = float(e @ e / (n - 3))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"beta": float(b[1]), "se": float(se[1]), "beta_z": float(b[2]), "se_z": float(se[2]),
            "resid_var": s2 * periods, "n": n}


def shrink_to_zero(b: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirisches Bayes mit Spike-and-Slab-Prior: Anteil π der Titel hat ein echtes β_z ~ N(0, τ²), der Rest 0.
    π und τ per Maximum-Likelihood aus dem Querschnitt (Gitter), dann Posterior-Mittel je Titel:
    E[β | β̂] = p_i · τ²/(τ² + SE²) · β̂ mit p_i = P(echter Effekt | β̂). Ein einzelner großer Effekt (Vonovia) bläht
    so nicht die Prior-Breite für alle auf (bei Normal-Prior würde Rauschen kaum geschrumpft). Rückgabe (β_post, w)."""
    b, se = np.asarray(b, float), np.asarray(se, float)
    ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
    post, w = np.zeros_like(b), np.zeros_like(b)
    if ok.sum() < 3:
        return post, w
    x, s2 = b[ok], se[ok] ** 2

    def dens(v):
        return np.exp(-0.5 * x ** 2 / v) / np.sqrt(2 * np.pi * v)
    best = (-np.inf, 0.0, 0.0)
    for tau in np.linspace(0.05, 8.0, 160):
        f1, f0 = dens(tau ** 2 + s2), dens(s2)
        for pi in np.linspace(0.0, 1.0, 101):
            ll = float(np.sum(np.log(pi * f1 + (1 - pi) * f0 + 1e-300)))
            if ll > best[0]:
                best = (ll, pi, tau)
    _, pi, tau = best
    f1, f0 = dens(tau ** 2 + s2), dens(s2)
    p_i = pi * f1 / (pi * f1 + (1 - pi) * f0 + 1e-300)
    w[ok] = p_i * tau ** 2 / (tau ** 2 + s2)
    post[ok] = w[ok] * x
    return post, w


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


def treynor_black_anchor(alpha: np.ndarray, resid_var: np.ndarray, budget: float, cash_floor: float = 0.02,
                         passiv: tuple[float, float] | None = None) -> np.ndarray:
    """Anker des robusten Kelly nach Treynor–Black: a_i ∝ α_i⁺ / σ²_ε,i, Cash als letzte Spalte; Summe 1.
    *passiv* = (ERP, σ²_Markt) p. a.: mit Index als Asset (vorletzte Spalte) echtes Treynor–Black (1973):
    aktiv a_i = (α_i⁺/σ²_ε,i) / (ERP/σ²_Markt), Rest des Budgets in den Index (aktiv höchstens das Budget)."""
    a = np.clip(np.nan_to_num(np.asarray(alpha, float), nan=0.0), 0.0, None)
    v = np.nan_to_num(np.asarray(resid_var, float), nan=np.inf)
    s = np.where(a > 0, a / np.maximum(v, 1e-6), 0.0)
    c = max(1.0 - budget, cash_floor)
    if passiv is not None:
        erp, var_m = passiv
        w = s / max(erp / var_m, 1e-9)
        if w.sum() > 1.0 - c:
            w *= (1.0 - c) / w.sum()
        return np.concatenate([w, [1.0 - c - w.sum(), c]])
    w = (1.0 - c) * s / s.sum() if s.sum() > 0 else np.zeros_like(s)
    return np.append(w, 1.0 - w.sum())
