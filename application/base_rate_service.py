"""
Innensicht gegen Außensicht (v3-Plan, Phase 2): Wo liegt eine Bewertung in den Basisraten, und wie breit sind
ihre Bänder im Vergleich zu dem, was Firmen mit ähnlichem Startpunkt tatsächlich erlebt haben?

Modellseite aus den Umsatz-/EBITDA-Pfaden der Engine (konsolidiert, 100 %):
  cagr5 = Umsatz-CAGR über 5 Jahre ab Basis; dm4 = Marge Jahr 5 − Marge Jahr 1 (Basismarge ist im Modell nicht separat).
Außensicht aus domain.base_rates, bedingt auf Branche, Größe und die Modell-Mediane von Jahr-1-Wachstum und -Marge.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from domain.base_rates import G_EDGES, M_EDGES, S_EDGES, bucket, conditional, outcomes, percentile_of, quantiles

H_GROWTH, H_MARGIN = 5, 4


@lru_cache(maxsize=4)
def _outcomes(h: int):
    from infrastructure.sec_xbrl import load_panel
    return outcomes(load_panel(), h)


def model_paths(results) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cagr5, dm4, g1, m1) je Sample aus den gespeicherten Segmentpfaden."""
    cf = list(results.segment_cashflows.values())
    h = min(H_GROWTH, *(d.forecast_years for d in cf))
    base = sum(d.base_revenue for d in cf)
    rev = lambda t: sum(d.revenue[:, t] for d in cf)           # noqa: E731
    ebitda = lambda t: sum(d.ebitda[:, t] for d in cf)         # noqa: E731
    cagr = (rev(h - 1) / base) ** (1 / h) - 1
    m1 = ebitda(0) / rev(0)
    dm = ebitda(h - 1) / rev(h - 1) - m1
    g1 = rev(0) / base - 1
    return cagr, dm, g1, m1


def compare(results, sektor: str | None, umsatz_usd: float, min_n: int = 200, ob=None) -> dict:
    """Vergleich Modell vs. Basisrate für Wachstum (cagr5) und Margenänderung (dm4).
    *ob* (optional): vorberechnete outcomes je Horizont {h: DataFrame} – für Tests ohne SEC-Daten."""
    cagr, dm, g1, m1 = model_paths(results)
    bed = dict(sektor=sektor, groesse=int(bucket([umsatz_usd], S_EDGES)[0]),
               g_start=int(bucket([np.median(g1)], G_EDGES)[0]), m_start=int(bucket([np.median(m1)], M_EDGES)[0]))
    out = {}
    for name, model, h, col in (("wachstum_5j", cagr, H_GROWTH, "cagr"), ("marge_4j", dm, H_MARGIN, "dm")):
        o = ob[h] if ob else _outcomes(h)
        vals, used = conditional(o, col, min_n=min_n, **bed)
        mq, bq = quantiles(model), quantiles(vals)
        out[name] = {"modell": mq, "basis": bq, "bedingungen": used,
                     "perzentil_modell_median": percentile_of(vals, mq["p50"]),
                     "breite_basis_zu_modell": (bq["p95"] - bq["p5"]) / max(mq["p95"] - mq["p5"], 1e-9)}
    return out
