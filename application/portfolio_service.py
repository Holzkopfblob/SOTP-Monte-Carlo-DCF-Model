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

from domain.irr import RAND_MAX_ANTEIL, hold_returns
from domain.portfolio import (
    correlate_samples, factor_tilt, horizon_returns, kelly_weights, market_world_returns, mix_worlds, orders, wealth,
)
from infrastructure.config_io import build_simulation_config
from infrastructure.monte_carlo_engine import MonteCarloEngine
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


def ev_inputs(sim_cfg) -> dict:
    """Für die Faktor-Korrektur: EBITDA Jahr 0 (Mitte, anteilig über die Quote) und Schulden (Bridge-Mitte +
    anteilige Tochterschulden). EV/EBITDA zum Markt = (Kurs × aktien + schulden) / ebitda. Dieselbe Definition
    nutzt der Querschnitt (Documents/KALIBRIERUNG-2026-11/querschnitt.py), der die Steigung schätzt."""
    segs = sim_cfg.segments
    return {
        "ebitda": sum(sg.base_revenue * sg.ebitda_margin.representative_value() * sg.ownership_share for sg in segs),
        "schulden": sim_cfg.corporate_bridge.net_debt
                    + sum(sg.subsidiary_net_debt.representative_value() * sg.ownership_share for sg in segs),
        "aktien": sim_cfg.corporate_bridge.shares_outstanding,
    }


def simulate_valuation(path: str | Path, shocks: dict | None = None, n: int | None = None) -> dict:
    """Bewertung per gespeichertem Seed neu simulieren (Samples je Aktie). *shocks*/*n*: gemeinsame
    Faktor-Szenarien (application/universe_simulation.py)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = cfg.get("metadata") or {}
    sim_cfg = build_simulation_config(cfg)
    if n:
        sim_cfg.n_simulations = n
    res = MonteCarloEngine(sim_cfg, shocks).run()
    samples = res.price_per_share
    stored = ((cfg.get("ergebnis") or {}).get("preis_je_aktie") or {}).get("p50")
    return {
        "samples": np.asarray(samples, float),
        "p50_neu": float(np.median(samples)),
        "p50_gespeichert": stored,
        "waehrung": meta.get("report_currency", "EUR"),
        "stichtag": meta.get("stichtag", ""),
        "kurs_recherche": meta.get("kurs_bei_recherche"),
        "stand": meta.get("stand", ""),
        **ev_inputs(sim_cfg),
        **({"results": res} if shocks is not None else {}),   # Faktor-Modus: IRR braucht die Cashflow-Pfade
    }


@dataclass
class Params:
    fraction: float = 0.5
    max_w: float = 0.35  # Notbremse (v3-Plan): Positionsgröße soll aus den Verteilungen kommen
    min_cash: float = 0.0   # Cash = Optimizer-Ergebnis; >0 nur als Untergrenze
    min_order_eur: float = 100.0
    allow_short: bool = False
    rf: float = 0.0315   # Bund 3 J. (Bundesbank-Zinsstruktur 13.11.2026); App lädt tagesaktuell
    horizon_years: int = 3
    seed: int = 42
    group_cap: float = 0.20  # max. Summe |Gewicht| je benannter Gruppe (z. B. „Auto“)
    p_market: float = 0.3   # Szenario-Mischung: Wahrscheinlichkeit „Markt hat recht“ (0 = Modell voll glauben)
    erp: float = 0.05       # Marktrendite in der Markt-Welt = rf + erp p. a.
    factor_shrink: float = 0.0  # Gegenprobe 16.11.: Faktor ist Telko-/Auto-Effekt, kein allgemeiner Stil ⇒ aus. Anteil des Value-Stils, der aus den Werten herausgenommen wird (0 = Modell voll, 1 = ganz)
    faktor_szenarien: bool = False  # v3 Phase 3: gemeinsame Cashflow-Schocks (Seite ruft dann simulate_universe)
    k_d: float = 0.035      # Fremdkapitalkosten nach Steuer für die Eigenkapital-IRR (≈ (rf + 1,5 pp) · 0,75)
    rho_world: float = 0.4  # Kopplung „Markt hat recht“ über Gruppen; 0,4 ≈ R² der Lücke auf das Marktmultiple
                            # (Querschnitt 16.11.2026, Documents/KALIBRIERUNG-2026-11/querschnitt.py)


@dataclass
class Result:
    table: pd.DataFrame
    wealth_ist: dict
    wealth_ziel: dict
    corr: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    wealth_samples: dict = field(default_factory=dict)   # {"Ist": array, "Ziel": array} in EUR (Stichprobe)
    steuer: dict = field(default_factory=dict)           # Verlusttopf vorher/Gewinne/nachher (EUR)


def _eur_closes(closes: pd.DataFrame, ticker: str, ccy: str) -> pd.Series:
    s = closes[ticker]
    if ccy == "USD":
        s = s / closes[FX_TICKER].ffill()
    return s


def rebalance(positions: list[dict], valuations: list[dict], closes: pd.DataFrame,
              cash_eur: float, p: Params, today: date | None = None, verlusttopf_eur: float = 0.0) -> Result:
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

    # Faktor-Szenarien (alle Bewertungen aus simulate_universe): Zeilen sind schon gemeinsame Welten ⇒ nichts
    # umsortieren, keine Kurs-Korrelation, keine Markt-Welt und keine Value-Korrektur obendrauf (Altpfad bis Abnahme).
    faktor = all("schock" in v for v in valuations)
    n = min(len(v["samples"]) for v in valuations)           # Samples sind iid → kürzen ok
    V = np.column_stack([v["samples"][:n] for v in valuations])
    ev_ebitda = np.array([(px * v["aktien"] + v["schulden"]) / v["ebitda"] if v.get("ebitda") else np.nan
                          for v, px in zip(valuations, px_local)])
    tilt = np.ones(len(names)) if faktor else factor_tilt(ev_ebitda, p.factor_shrink)
    if not faktor:
        V = correlate_samples(V, corr.values, seed=p.seed) * tilt
    p_wert = (V > px_local).mean(axis=0)
    rf_h = (1 + p.rf) ** p.horizon_years - 1
    sigma = (rets.std() * np.sqrt(52)).reindex(names).fillna(0.30).clip(0.10, 1.0).to_numpy()
    rand = np.zeros(len(names), bool)
    if faktor:
        # v3 Phase 4: Halte-Rendite aus der impliziten Rendite zum Kurs (domain/irr.hold_returns) statt Wert/Kurs − 1
        hr = [hold_returns(v["results"], px, p.k_d, p.horizon_years) for v, px in zip(valuations, px_local)]
        R_model = np.column_stack([x["R"][:n] for x in hr])
        rand = np.array([x["am_rand"] > RAND_MAX_ANTEIL for x in hr])
        if rand.any():   # Bridge > Börsenwert: IRR unbrauchbar ⇒ Marktrendite mit Kursvola, Gewicht eingefroren
            R_mkt = market_world_returns(n, sigma, corr.values, p.rf, p.erp, p.horizon_years, seed=p.seed + 1)
            R_model[:, rand] = R_mkt[:, rand]
            warnings.append("IRR nicht aussagekräftig (Bridge-Posten größer als der Börsenwert, Bewertung prüfen), "
                            "Gewicht bleibt: " + ", ".join(np.array(names)[rand]) + ".")
    else:
        R_model = horizon_returns(V, px_local)               # Wert und Kurs in gleicher Währung

    # Szenario-Mischung: Welt „Markt hat recht“ (Kurs fair, Marktrendite rf + ERP, Volatilität aus der
    # Kurshistorie) mit Wahrscheinlichkeit p_market je Gruppe; Titel einer Gruppe wechseln gemeinsam.
    groups = [str(pos.get("gruppe") or "").strip() or pos["firma"] for pos in positions]
    if p.p_market > 0 and not faktor:
        R_mkt = market_world_returns(n, sigma, corr.values, p.rf, p.erp, p.horizon_years, seed=p.seed + 1)
        R, _ = mix_worlds(R_model, R_mkt, p.p_market, groups, seed=p.seed + 2, rho=p.rho_world)
    else:
        R = R_model

    shares = np.array([float(pos.get("stueck") or 0) for pos in positions])
    total = cash_eur + float(shares @ px_eur)
    w_ist = shares * px_eur / total if total > 0 else np.zeros(len(names))

    # Optimieren → Orders → Titel, deren Order an der Mindestorder scheitert, auf Ist-Gewicht
    # einfrieren und das tatsächlich verfügbare Geld neu verteilen, bis jede Order ausführbar ist.
    lo = np.full(len(names), -p.max_w if p.allow_short else 0.0)
    hi = np.full(len(names), p.max_w)
    frozen = rand.copy()
    lo[frozen] = hi[frozen] = w_ist[frozen]
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
    if (frozen & ~rand).any():
        warnings.append("Wegen der Mindestorder unverändert gelassen und neu verteilt: "
                        + ", ".join(np.array(names)[frozen & ~rand]) + ".")

    for name, val, px in zip(names, valuations, px_local):
        stored = val["p50_gespeichert"]
        if stored and not faktor and abs(val["p50_neu"] / stored - 1) > REPRO_TOL:
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
        "EV/EBITDA": ev_ebitda,
        "Faktor ×": tilt,
        "P(Wert>Kurs)": p_wert,
        **{col: np.where(rand, np.nan, [x[k] for x in hr]) if faktor else np.nan       # am Rand: IRR ohne Aussage
           for col, k in (("IRR P50", "irr"), ("IRR EK", "irr_ek"), ("D/E", "d_e"))},
        "E[R] Horizont": R.mean(axis=0),
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

    # Steuer (Aktienverlusttopf): Veräußerungsgewinne der Verkäufe verbrauchen den Topf, Verluste füllen ihn.
    # Dividenden verrechnet der Aktienverlusttopf nicht. ponytail: Wert des Steueraufschubs fehlt, solange der Topf
    # die Gewinne deckt (Depot ≪ Topf); nötig, sobald Gewinne den Topf erreichen.
    einstand = np.array([float(pos.get("einstand_eur") or np.nan) for pos in positions])
    gewinn = np.where(delta < 0, (px_eur - einstand) * -delta, 0.0)
    table.insert(table.columns.get_loc("Order €") + 1, "Gewinn Verkauf €",
                 np.r_[np.where(delta < 0, gewinn, np.nan), np.nan])
    g_sum = float(np.nansum(gewinn))
    steuer = {"topf_vorher": verlusttopf_eur, "gewinne": g_sum, "topf_nachher": verlusttopf_eur - g_sum}
    if np.isnan(gewinn[delta < 0]).any():
        warnings.append("Einstand fehlt bei Verkäufen: " + ", ".join(np.array(names)[(delta < 0) & np.isnan(einstand)])
                        + " – Topf-Verbrauch unvollständig.")
    if g_sum > verlusttopf_eur:
        warnings.append(f"Verkaufsgewinne {g_sum:,.0f} € übersteigen den Verlusttopf {verlusttopf_eur:,.0f} € – "
                        f"≈ {0.26375 * (g_sum - verlusttopf_eur):,.0f} € Steuer (KapESt + Soli).")

    W = {"Ist": wealth(R, w_ist, rf_h) * total, "Ziel": wealth(R, w, rf_h) * total}

    def stats(arr):
        return {"p5": float(np.percentile(arr, 5)), "p50": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)), "mean": float(arr.mean()), "start": total}

    sample = {k: v[:: max(1, len(v) // 5000)] for k, v in W.items()}   # für Diagramme
    return Result(table, stats(W["Ist"]), stats(W["Ziel"]), corr, warnings, sample, steuer)
