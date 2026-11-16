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

from application.base_rate_service import H_GROWTH, H_MARGIN, _outcomes
from domain.base_rates import G_EDGES, M_EDGES, S_EDGES, bucket, conditional, sektor as sektor_of
from domain.factor_model import common_sd, draw_shocks, firm_params
from infrastructure.config_io import build_simulation_config

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


def shock_params(name: str, sim_cfg, sektor: str | None, ccy: str, ob: dict | None = None,
                 common: dict | None = None) -> dict:
    """Schock-Parameter einer Firma; *ob*/*common* nur für Tests ohne SEC-Daten."""
    bed = start_conditions(sim_cfg, sektor, ccy)
    o = (lambda h: ob[h]) if ob else _outcomes
    g1 = np.log1p(conditional(o(1), "cagr", **bed)[0])
    gh = H_GROWTH * np.log1p(conditional(o(H_GROWTH), "cagr", **bed)[0])
    m1 = conditional(o(1), "dm", **bed)[0]
    mh = conditional(o(H_MARGIN), "dm", **bed)[0]
    return firm_params(name, sektor or OHNE_BRANCHE, g1, gh, H_GROWTH, m1, mh, H_MARGIN, common or _common()) \
        | {"bedingungen": bed}


def simulate_universe(items: list[tuple[str, str | Path, str | None]], n: int = 20_000, seed: int = 0,
                      scale: float = 1.0) -> list[dict]:
    """*items*: [(Firma, Bewertungsdatei, Branche)]. *scale*: Gewicht der Außensicht (1 = volle Basisraten-Breite)."""
    from application.portfolio_service import simulate_valuation   # Kreisimport vermeiden
    cfgs = [json.loads(Path(p).read_text(encoding="utf-8")) for _, p, _ in items]
    sims = [build_simulation_config(c) for c in cfgs]
    params = [shock_params(f, sc, s, (c.get("metadata") or {}).get("report_currency", "EUR"))
              for (f, _, s), sc, c in zip(items, sims, cfgs)]
    T = max(sg.forecast_years for sc in sims for sg in sc.segments)
    shocks = draw_shocks(params, n, T, seed, scale)
    return [simulate_valuation(p, sh, n) | {"schock": prm} for (_, p, _), sh, prm in zip(items, shocks, params)]
