"""
Portfolio-Service: Depot laden/speichern, Bewertungen finden und neu simulieren,
Zielgewichte und Orders berechnen.

depot.json (liegt außerhalb des Repos, z. B. in 05_PRIVAT/Finanzen):
  {"bewertungen_ordner": "...", "cash_eur": 5000,
   "positionen": [{"firma": "Rheinmetall AG", "ticker": "RHM.DE", "stueck": 3, "datei": "",
                     "einstand_eur": 812.40, "kaufdatum": "2026-10-31"}]}
Bruchstücke erlaubt; Einstand/Kaufdatum werden nur mitgeführt (Steuer später, E24).
``datei`` leer = neueste Bewertung der Firma im Ordner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from application.simulation_service import SimulationService
from domain.portfolio import (
    correlate_samples, horizon_returns, kelly_weights, market_world_returns, mix_worlds, orders, wealth,
)
from infrastructure.config_io import build_simulation_config
from infrastructure.market_data import FX_TICKER

SUPPORTED_CCY = {"EUR", "USD"}  # ponytail: nur EUR/USD; weitere Währungen = weiteres FX-Paar laden
REPRO_TOL = 0.005               # Neulauf-P50 darf 0,5 % vom gespeicherten abweichen
STALE_DAYS = 30


def load_depot(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"bewertungen_ordner": str(p.parent / "SOTP-Monte-Carlo-DCF_Bewertungen"),
                "cash_eur": 0.0, "positionen": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_depot(path: str | Path, depot: dict) -> None:
    Path(path).write_text(json.dumps(depot, indent=2, ensure_ascii=False), encoding="utf-8")


def find_valuations(folder: str | Path) -> dict[str, list[Path]]:
    """Firma → Bewertungsdateien, älteste zuerst (stichtag, dann saved_at)."""
    out: dict[str, list[tuple[tuple[str, str], Path]]] = {}
    for f in Path(folder).glob("*.json"):
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        meta = cfg.get("metadata") or {}
        if not meta.get("company") or not cfg.get("ergebnis"):
            continue
        key = (str(meta.get("stichtag", "")), str(cfg.get("saved_at", "")))
        out.setdefault(meta["company"], []).append((key, f))
    return {c: [f for _, f in sorted(v)] for c, v in out.items()}


def simulate_valuation(path: str | Path) -> dict:
    """Bewertung per gespeichertem Seed neu simulieren (Samples je Aktie)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = cfg.get("metadata") or {}
    samples = SimulationService.run_simulation(build_simulation_config(cfg)).price_per_share
    stored = ((cfg.get("ergebnis") or {}).get("preis_je_aktie") or {}).get("p50")
    return {
        "samples": np.asarray(samples, float),
        "p50_neu": float(np.median(samples)),
        "p50_gespeichert": stored,
        "waehrung": meta.get("report_currency", "EUR"),
        "stichtag": meta.get("stichtag", ""),
        "kurs_recherche": meta.get("kurs_bei_recherche"),
        "stand": meta.get("stand", ""),
    }


@dataclass
class Params:
    fraction: float = 0.5
    max_w: float = 0.15
    min_cash: float = 0.0   # Cash = Optimizer-Ergebnis; >0 nur als Untergrenze
    min_order_eur: float = 100.0
    allow_short: bool = False
    rf: float = 0.0315   # Bund 3 J. (Bundesbank-Zinsstruktur 13.11.2026); App lädt tagesaktuell
    horizon_years: int = 3
    seed: int = 42
    group_cap: float = 0.20  # max. Summe |Gewicht| je benannter Gruppe (z. B. „Auto“)
    p_market: float = 0.3   # Szenario-Mischung: Wahrscheinlichkeit „Markt hat recht“ (0 = Modell voll glauben)
    erp: float = 0.05       # Marktrendite in der Markt-Welt = rf + erp p. a.


@dataclass
class Result:
    table: pd.DataFrame
    wealth_ist: dict
    wealth_ziel: dict
    corr: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    wealth_samples: dict = field(default_factory=dict)   # {"Ist": array, "Ziel": array} in EUR (Stichprobe)


def _eur_closes(closes: pd.DataFrame, ticker: str, ccy: str) -> pd.Series:
    s = closes[ticker]
    if ccy == "USD":
        s = s / closes[FX_TICKER].ffill()
    return s


def rebalance(positions: list[dict], valuations: list[dict], closes: pd.DataFrame,
              cash_eur: float, p: Params, today: date | None = None) -> Result:
    """positions[i] (firma, ticker, stueck) gehört zu valuations[i] (simulate_valuation)."""
    today = today or date.today()
    warnings: list[str] = []
    names = [pos["firma"] for pos in positions]

    eur = {}
    for pos, val in zip(positions, valuations):
        if val["waehrung"] not in SUPPORTED_CCY:
            raise ValueError(f"{pos['firma']}: Währung {val['waehrung']} nicht unterstützt")
        eur[pos["firma"]] = _eur_closes(closes, pos["ticker"], val["waehrung"])
    eur_df = pd.DataFrame(eur)
    fx_now = float(closes[FX_TICKER].dropna().iloc[-1])

    px_local = np.array([float(closes[pos["ticker"]].dropna().iloc[-1]) for pos in positions])
    to_eur = np.array([1 / fx_now if v["waehrung"] == "USD" else 1.0 for v in valuations])
    px_eur = px_local * to_eur

    # Korrelation: Spearman auf Wochen-Log-Renditen in EUR (Tagesdaten unterschätzen sie
    # wegen versetzter Börsenschlüsse US/EU), paarweise → kürzere Historie (SPCX) möglich
    rets = np.log(eur_df.resample("W-FRI").last()).diff()
    corr = rets.corr(method="spearman", min_periods=20).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    for name in names:
        n_obs = int(rets[name].count())
        if n_obs < 52 * 3 * 0.9:
            warnings.append(f"{name}: nur {n_obs} Wochen Kurshistorie für die Korrelation"
                            + (" (zu kurz, Korrelation 0 angenommen)." if n_obs < 20 else "."))

    n = min(len(v["samples"]) for v in valuations)           # Samples sind iid → kürzen ok
    V = np.column_stack([v["samples"][:n] for v in valuations])
    V = correlate_samples(V, corr.values, seed=p.seed)
    R_model = horizon_returns(V, px_local)                   # Wert und Kurs in gleicher Währung
    rf_h = (1 + p.rf) ** p.horizon_years - 1

    # Szenario-Mischung: Welt „Markt hat recht“ (Kurs fair, Marktrendite rf + ERP, Volatilität aus der
    # Kurshistorie) mit Wahrscheinlichkeit p_market je Gruppe; Titel einer Gruppe wechseln gemeinsam.
    sigma = (rets.std() * np.sqrt(52)).reindex(names).fillna(0.30).clip(0.10, 1.0).to_numpy()
    groups = [str(pos.get("gruppe") or "").strip() or pos["firma"] for pos in positions]
    if p.p_market > 0:
        R_mkt = market_world_returns(n, sigma, corr.values, p.rf, p.erp, p.horizon_years, seed=p.seed + 1)
        R, _ = mix_worlds(R_model, R_mkt, p.p_market, groups, seed=p.seed + 2)
    else:
        R = R_model

    shares = np.array([float(pos.get("stueck") or 0) for pos in positions])
    total = cash_eur + float(shares @ px_eur)
    w_ist = shares * px_eur / total if total > 0 else np.zeros(len(names))

    # Optimieren → Orders → Titel, deren Order an der Mindestorder scheitert, auf Ist-Gewicht
    # einfrieren und das tatsächlich verfügbare Geld neu verteilen, bis jede Order ausführbar ist.
    lo = np.full(len(names), -p.max_w if p.allow_short else 0.0)
    hi = np.full(len(names), p.max_w)
    frozen = np.zeros(len(names), bool)
    named = {g for g in groups if groups.count(g) > 1}
    caps = [([i for i, g in enumerate(groups) if g == name], p.group_cap) for name in sorted(named)]
    for _ in range(len(names) + 1):
        w = kelly_weights(R, rf_h, p.fraction, p.max_w, p.min_cash, p.allow_short, lo, hi, caps)
        delta, target = orders(shares, px_eur, cash_eur, w, p.min_order_eur)
        new = (delta == 0) & ~np.isclose(w, w_ist, atol=1e-4) & ~frozen
        if not new.any():
            break
        frozen |= new
        lo[frozen] = hi[frozen] = w_ist[frozen]
    if frozen.any():
        warnings.append("Wegen der Mindestorder unverändert gelassen und neu verteilt: "
                        + ", ".join(np.array(names)[frozen]) + ".")

    for name, val, px in zip(names, valuations, px_local):
        stored = val["p50_gespeichert"]
        if stored and abs(val["p50_neu"] / stored - 1) > REPRO_TOL:
            warnings.append(f"{name}: Neulauf-P50 {val['p50_neu']:.2f} weicht vom gespeicherten "
                            f"{stored:.2f} ab (Engine geändert?).")
        if val["kurs_recherche"] and not 0.5 < px / val["kurs_recherche"] < 2:
            warnings.append(f"{name}: Kurs {px:.2f} passt nicht zum Recherchekurs "
                            f"{val['kurs_recherche']:.2f} – Ticker/Währung prüfen.")
        try:
            age = (today - date.fromisoformat(val["stichtag"])).days
            if age > STALE_DAYS:
                warnings.append(f"{name}: Bewertung ist {age} Tage alt.")
        except ValueError:
            warnings.append(f"{name}: Stichtag fehlt.")

    table = pd.DataFrame({
        "Firma": names,
        "Stand": [v["stand"] for v in valuations],
        "Kurs €": px_eur,
        "Wert P50 €": [v["p50_neu"] * f for v, f in zip(valuations, to_eur)],
        "Kurs lokal": [f"{px:.2f} {v['waehrung']}" for v, px in zip(valuations, px_local)],
        "Upside P50": [v["p50_neu"] / px - 1 for v, px in zip(valuations, px_local)],
        "P(Wert>Kurs)": (R_model > 0).mean(axis=0),
        "E[R] 3J (Mix)": R.mean(axis=0),
        "Vola p. a.": sigma,
        "Gruppe": groups,
        "Ist-Gewicht": w_ist,
        "Ziel-Gewicht": w,
        "Δ Gewicht": w - w_ist,
        "Order Stück": delta,
        "Order €": delta * px_eur,
        "Ziel Stück": target,
    })
    # Cash-Zeile: Ziel = was der Optimizer übrig lässt; Order € = Cash-Veränderung durch die Orders
    cash_order = -float((delta * px_eur).sum())
    table.loc[len(table)] = {
        "Firma": "Cash (EUR)", "Ist-Gewicht": cash_eur / total if total > 0 else 0.0,
        "Ziel-Gewicht": 1.0 - float(w.sum()),
        "Δ Gewicht": 1.0 - float(w.sum()) - (cash_eur / total if total > 0 else 0.0),
        "Order €": cash_order,
    }

    W = {"Ist": wealth(R, w_ist, rf_h) * total, "Ziel": wealth(R, w, rf_h) * total}

    def stats(arr):
        return {"p5": float(np.percentile(arr, 5)), "p50": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)), "mean": float(arr.mean()), "start": total}

    sample = {k: v[:: max(1, len(v) // 5000)] for k, v in W.items()}   # für Diagramme
    return Result(table, stats(W["Ist"]), stats(W["Ziel"]), corr, warnings, sample)
