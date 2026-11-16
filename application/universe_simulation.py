"""
Gemeinsame Szenarien über alle Titel (v3-Plan Phase 3).

Jede Bewertung läuft mit denselben n Zeilen: Zeile i ist für alle Firmen dieselbe Welt (Makro- und Branchenpfad
aus domain/factor_model). Die Schockbreite je Firma kommt aus ihrer Basisrate (Branche, Größe, Startwachstum und
Startmarge wie Check 12). Ergebnis: dieselben Dicts wie portfolio_service.simulate_valuation, Spalten bereits
gemeinsam verteilt – keine Rang-Umsortierung (Iman-Conover) und keine Kurshistorie-Korrelation obendrauf.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from application.base_rate_service import H_GROWTH, H_MARGIN, _outcomes, model_paths
from domain.base_rates import G_EDGES, M_EDGES, S_EDGES, bucket, conditional, sektor as sektor_of
from domain.factor_model import center_drift, common_sd, draw_shocks, firm_params
from infrastructure.config_io import build_simulation_config
from infrastructure.monte_carlo_engine import MonteCarloEngine

USD_JE = {"USD": 1.0, "EUR": 1.105}   # ponytail: fester Kurs, reicht für Größenklassen (Faktor 10)
OHNE_BRANCHE = "—"                    # kein Branchenfaktor (eigener Strom, Anteil 0)


@lru_cache(maxsize=1)
def _common() -> dict:
    from infrastructure.sec_xbrl import load_panel
    return common_sd(load_panel(), sektor_of)


def start_conditions(sim_cfg, sektor: str | None, ccy: str) -> dict:
    """Bedingungen der Basisrate aus der Konfiguration (umsatzgewichtete Mitten von Wachstum und Marge)."""
    segs = sim_cfg.segments
    rev = sum(s.base_revenue for s in segs)
    g = sum(s.base_revenue * s.revenue_growth.representative_value() for s in segs) / rev
    m = sum(s.base_revenue * s.ebitda_margin.representative_value() for s in segs) / rev
    return {"sektor": sektor, "groesse": int(bucket([rev * 1e6 * USD_JE.get(ccy, 1.0)], S_EDGES)[0]),
            "g_start": int(bucket([g], G_EDGES)[0]), "m_start": int(bucket([m], M_EDGES)[0])}


MIN_N_SCHOCK = 100   # IQR ist ab ~100 Fällen stabil; kleiner als bei Check 12, damit die Größenklasse öfter bleibt
# Für die Schockbreite zählt die Größe mehr als die Branche (Panel: σ 0,17 / 0,11 / 0,08 je Größenklasse) ⇒ Branche
# vor Größe lockern (Check 12 behält LOCKERUNG, dort geht es um die Lage im Branchenvergleich).
LOCKERUNG_SCHOCK = ["g_start", "m_start", "sektor", "groesse"]
# Für das Zentrum zählt das Startwachstum am meisten (Mean Reversion hoher Wachstumsraten) ⇒ zuletzt lockern; sonst
# bekäme NVIDIA (36 % p. a. im Modell) den Median gewöhnlicher großer Halbleiterfirmen (1,6 %).
LOCKERUNG_ZENTRUM = ["m_start", "sektor", "groesse", "g_start"]
ZENTRIERUNG = 0.5    # Nutzerentscheid 16.11.2026: Umsatzpfad zur Hälfte Richtung Basisrate (0 = Analyst, 1 = Basisrate)


def _cohort(o, col: str, h: int, bed: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Beide Momente aus derselben Kohorte: erst den langen Horizont (weniger Fälle) lockern, dann den 1-J.-Horizont
    mit genau diesen Bedingungen. Sonst stammen σ_1 und σ_h aus verschiedenen Gruppen und φ (ihr Verhältnis) kippt."""
    xh, used = conditional(o(h), col, min_n=MIN_N_SCHOCK, lockerung=LOCKERUNG_SCHOCK, **bed)
    x1, _ = conditional(o(1), col, min_n=0, **used)
    return x1, xh, used


def shock_params(name: str, sim_cfg, sektor: str | None, ccy: str, ob: dict | None = None,
                 common: dict | None = None) -> dict:
    """Schock-Parameter einer Firma; *ob*/*common* nur für Tests ohne SEC-Daten."""
    bed = start_conditions(sim_cfg, sektor, ccy)
    o = (lambda h: ob[h]) if ob else _outcomes
    g1, gh, used_g = _cohort(o, "cagr", H_GROWTH, bed)
    m1, mh, used_m = _cohort(o, "dm", H_MARGIN, bed)
    # ponytail: Basisrate enthält Zukäufe (Wachstum eher überzeichnet) und nur das Wachstum wird zentriert, nicht die Marge
    ziel, used_z = conditional(o(H_GROWTH), "cagr", min_n=MIN_N_SCHOCK, lockerung=LOCKERUNG_ZENTRUM, **bed)
    return firm_params(name, sektor or OHNE_BRANCHE, np.log1p(g1), H_GROWTH * np.log1p(gh), H_GROWTH,
                       m1, mh, H_MARGIN, common or _common()) | {
        "bedingungen": {"g": used_g, "m": used_m, "zentrum": used_z}, "ziel_cagr5": float(np.median(ziel))}


def simulate_universe(items: list[tuple[str, str | Path, str | None]], n: int = 20_000, seed: int = 0,
                      scale: float = 1.0, zentrierung: float = ZENTRIERUNG) -> list[dict]:
    """*items*: [(Firma, Bewertungsdatei, Branche)]. *scale*: Gewicht der Außensicht (1 = volle Basisraten-Breite).
    *zentrierung*: Anteil, um den der Median-Umsatzpfad (5 J.) Richtung Basisrate rückt."""
    from application.portfolio_service import simulate_valuation   # Kreisimport vermeiden
    cfgs = [json.loads(Path(p).read_text(encoding="utf-8")) for _, p, _ in items]
    sims = [build_simulation_config(c) for c in cfgs]
    params = [shock_params(f, sc, s, (c.get("metadata") or {}).get("report_currency", "EUR"))
              for (f, _, s), sc, c in zip(items, sims, cfgs)]
    T = max(sg.forecast_years for sc in sims for sg in sc.segments)
    shocks = draw_shocks(params, n, T, seed, scale)
    for sc, prm, sh in zip(sims, params, shocks):
        h = min(H_GROWTH, *(sg.forecast_years for sg in sc.segments))
        n0, sc.n_simulations = sc.n_simulations, 500
        prm["modell_cagr5"] = float(np.median(model_paths(MonteCarloEngine(sc).run())[0]))
        sc.n_simulations = n0
        sh["g"] = center_drift(sh["g"], prm["modell_cagr5"], prm["ziel_cagr5"], zentrierung, h)
        prm["zentrierung"] = zentrierung
    return [simulate_valuation(p, sh, n) | {"schock": prm} for (_, p, _), sh, prm in zip(items, shocks, params)]
