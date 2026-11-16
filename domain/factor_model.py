"""
Faktormodell für Cashflow-Schocks (v3-Plan Phase 3): ehrliche Breite aus den Basisraten, gemeinsame Szenarien über Firmen.

Je Firma zwei Schockpfade über T Jahre, die auf die Analysten-Pfade der Engine wirken:
  Wachstum  s_t = φ_g · s_{t−1} + η_t   (Log-Wachstum; Summe = dauerhafte Abweichung des Umsatzniveaus)
  Marge     e_t = φ_m · e_{t−1} + ν_t   (Niveau der EBITDA-Marge in Punkten)
mit s_0 = e_0 = 0 (Startpunkt bekannt). Die Innovationen zerlegen sich in Makro (alle Firmen), Branche und Firma:
  η_t = σ · (√β_makro · ξ_t + √β_branche · ψ_{Branche,t} + √(1 − β_makro − β_branche) · ε_t).
σ und φ je Firma passen zwei Momente der Basisrate (gleiche Bedingungen wie Check 12): Streuung über 1 Jahr und über
h Jahre. Die β sind Anteile der Jahres-Streuung, die im Panel auf Jahr bzw. Branche×Jahr entfallen.
Streuung robust als IQR/1,349: Übernahmen erzeugen Sprünge in den Rändern, die das Maß kaum bewegen.

SEC-Panel 2009–2025, große Firmen (Documents/KALIBRIERUNG-2026-11/faktoren_panel.py): Wachstum φ ≈ 0,43 trifft die
Niveau-Streuung bis 10 J. (0,46 vs. 0,46), Marge φ ≈ 0,96 (fast dauerhaft).
# ponytail: Modellfehler je Branche (gemeinsamer Bias) fehlt – Rücktest Feb. (10 Firmen) zeigt keinen; ergänzen, wenn
# realisierungen.csv je Branche genug Fälle hat.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

MAX_BETA = 0.9          # Firmenanteil ≥ 10 %, sonst wären Branchen-Titel austauschbar


def robust_sd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float((np.percentile(x, 75) - np.percentile(x, 25)) / 1.349) if len(x) else float("nan")


def growth_ratio(phi: float, h: int) -> float:
    """Var(Σ_{t≤h} s_t) / σ² für s_t = φ s_{t−1} + η_t, s_0 = 0 (h = 1 ⇒ 1, φ = 1 ⇒ Σ k²)."""
    return float(sum(sum(phi ** j for j in range(h - k + 1)) ** 2 for k in range(1, h + 1)))


def margin_ratio(phi: float, h: int) -> float:
    """Var(e_h) / σ² für e_t = φ e_{t−1} + ν_t, e_0 = 0."""
    return float(sum(phi ** (2 * j) for j in range(h)))


def fit_phi(ratio_fn, target: float, h: int) -> float:
    """φ ∈ [0, 1] mit ratio_fn(φ, h) = target (monoton steigend); außerhalb ⇒ Rand."""
    lo, hi = 0.0, 1.0
    if not np.isfinite(target) or target <= ratio_fn(lo, h):
        return lo
    if target >= ratio_fn(hi, h):
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ratio_fn(mid, h) < target else (lo, mid)
    return (lo + hi) / 2


def common_sd(panel: pd.DataFrame, sektor_of) -> dict:
    """Jahres-Streuung der gemeinsamen Faktoren aus dem Panel (cik, jahr, umsatz, ebit, da, sic):
    {"g"|"m": {"makro": sd, <Branche>: sd}} – Makro = Median je Jahr, Branche = Median je Branche×Jahr minus Makro.
    g = Log-Umsatzwachstum, m = Margenänderung (EBITDA/Umsatz) ggü. Vorjahr."""
    p = panel.sort_values(["cik", "jahr"]).copy()
    p["sektor"] = p.sic.map(sektor_of)
    p = p[(p.sektor != "Finanz") & (p.umsatz >= 50e6)]
    prev = p.groupby("cik")
    with np.errstate(divide="ignore", invalid="ignore"):
        p["g"] = np.log(p.umsatz / prev.umsatz.shift(1))
    marge = ((p.ebit + p.da) / p.umsatz).where(lambda m: m.abs() <= 1)
    p["m"] = marge - marge.groupby(p.cik).shift(1)
    p = p[p.jahr == prev.jahr.shift(1) + 1]                       # nur lückenlose Folgejahre
    out = {}
    for col in ("g", "m"):
        q = p[np.isfinite(p[col])]
        mak = q.groupby("jahr")[col].median()
        sy = q.groupby(["sektor", "jahr"])[col].median().reset_index()
        sy[col] -= sy.jahr.map(mak)
        out[col] = {"makro": float(mak.std()), **sy.groupby("sektor")[col].std().fillna(0.0).to_dict()}
    return out


def firm_params(name: str, sektor: str, g1: np.ndarray, gh: np.ndarray, h_g: int,
                m1: np.ndarray, mh: np.ndarray, h_m: int, common: dict) -> dict:
    """Schock-Parameter einer Firma aus ihren Basisraten: g1/gh = Log-Umsatzänderung über 1 bzw. h_g Jahre,
    m1/mh = Margenänderung über 1 bzw. h_m Jahre (jeweils Werte der bedingten Basisrate)."""
    out = {"name": name, "sektor": sektor}
    for k, x1, xh, h, ratio in (("g", g1, gh, h_g, growth_ratio), ("m", m1, mh, h_m, margin_ratio)):
        s1, sh = robust_sd(x1), robust_sd(xh)
        b_mak = min((common[k]["makro"] / s1) ** 2, MAX_BETA)
        b_sek = min((common[k].get(sektor, 0.0) / s1) ** 2, MAX_BETA - b_mak)
        out[k] = {"sigma": s1, "phi": fit_phi(ratio, (sh / s1) ** 2, h), "beta": (b_mak, b_sek)}
    return out


def _ar(eta: np.ndarray, phi: float) -> np.ndarray:
    s, prev = np.empty_like(eta), np.zeros(eta.shape[0])
    for t in range(eta.shape[1]):
        prev = phi * prev + eta[:, t]
        s[:, t] = prev
    return s


def _rng(seed: int, key: str) -> np.random.Generator:
    """Eigener Strom je Faktor/Firma, unabhängig von der Reihenfolge der Titel."""
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def draw_shocks(firms: list[dict], n: int, T: int, seed: int = 0, scale: float = 1.0) -> list[dict]:
    """Gemeinsame Ziehung: Zeile i ist für alle Firmen dieselbe Welt (Makro- und Branchenpfad).
    *scale* = Gewicht der Außensicht (1 = volle Basisraten-Breite). Rückgabe je Firma {"g", "m": (n, T), "m_phi"}."""
    out = []
    for f in firms:
        res = {"m_phi": f["m"]["phi"]}
        for k in ("g", "m"):
            p = f[k]
            b_mak, b_sek = p["beta"]
            z = (np.sqrt(b_mak) * _rng(seed, f"makro|{k}").standard_normal((n, T))
                 + np.sqrt(b_sek) * _rng(seed, f"branche|{f['sektor']}|{k}").standard_normal((n, T))
                 + np.sqrt(1 - b_mak - b_sek) * _rng(seed, f"firma|{f['name']}|{k}").standard_normal((n, T)))
            res[k] = _ar(scale * p["sigma"] * z, p["phi"])
        out.append(res)
    return out


def center_drift(g: np.ndarray, model_cagr: float, target_cagr: float, lam: float, h: int) -> np.ndarray:
    """Zentrum des Umsatzpfads um *lam* (0…1) Richtung Basisrate schieben: konstante Log-Drift in den ersten *h* Jahren,
    sodass die h-J.-CAGR des Medians von model_cagr auf (1+m)^(1−λ)·(1+z)^λ − 1 rückt. Das Niveau bleibt danach
    verschoben (Schock-Pfade sind kumuliert). Innensicht und Außensicht gemischt statt nur verbreitert."""
    out = g.copy()
    out[:, :h] += lam * (np.log1p(target_cagr) - np.log1p(model_cagr))
    return out
