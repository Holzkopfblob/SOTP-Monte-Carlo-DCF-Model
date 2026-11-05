"""
Plausibility checks for a valuation (Checks 4, 5, 6, 7, 9, 11 of the research prompt).

Pure functions on ``SimulationConfig`` (+ optional results and metadata), so the
app and headless scripts show the same verdicts. Thresholds are module constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.fade import build_fade_curve
from domain.models import RevenueGrowthMode, TerminalValueMethod
from domain.valuation import compute_fcff_vectors

OK, WARN, FAIL, INFO = "✅", "⚠️", "❌", "ℹ️"

MIN_WACC_TVG_SPREAD = 0.02      # Check 4: WACC − TV-g below 2 pp → warning
TV_EV_WARN = 0.70               # Check 5: same red line as the TV/EV section
FCFF_TOL_OK, FCFF_TOL_WARN = 0.10, 0.25   # Check 11: |model/ist − 1|
FAIR_VALUE_GAP_WARN = 0.50      # Check 9: |P50/Kurs − 1| above 50 % → explain the gap


@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str


def _rep(d):
    return None if d is None else float(d.representative_value())


def _year1_params(seg) -> dict[str, float]:
    """Year-1 values as the engine uses them (parameter fade already applied)."""
    out = {}
    for p, init, term in [
        ("em", seg.ebitda_margin, seg.ebitda_margin_terminal),
        ("da", seg.da_pct_revenue, seg.da_pct_revenue_terminal),
        ("tx", seg.tax_rate, seg.tax_rate_terminal),
        ("cx", seg.capex_pct_revenue, seg.capex_pct_revenue_terminal),
        ("nwc", seg.nwc_pct_delta_revenue, seg.nwc_pct_delta_revenue_terminal),
    ]:
        v = np.array([_rep(init)])
        if term is not None and seg.revenue_growth_mode == RevenueGrowthMode.FADE:
            v = build_fade_curve(v, np.array([_rep(term)]), seg.forecast_years, seg.fade_speed)
        out[p] = v
    return out


def year1_fcff(seg) -> float:
    """Deterministic year-1 FCFF from representative values (same chain as the engine)."""
    p = _year1_params(seg)
    fcff, _, _ = compute_fcff_vectors(
        seg.base_revenue, seg.forecast_years, np.array([_rep(seg.revenue_growth)]),
        p["em"], p["da"], p["tx"], p["cx"], p["nwc"],
        growth_mode=seg.revenue_growth_mode,
        terminal_growth=np.array([_rep(seg.terminal_growth_rate)]),
        fade_speed=seg.fade_speed,
    )
    return float(fcff[0, 0])


def roic_is_artifact(seg) -> bool:
    """Implied ROIC = g × NOPAT / reinvestment is meaningless for g ≤ 0 or reinvestment ≤ 0.

    The engine computes it from the initial growth and year-1 parameters, so the
    rule uses the same inputs (representative values).
    """
    g = _rep(seg.revenue_growth)
    reinvest = (_rep(seg.capex_pct_revenue) - _rep(seg.da_pct_revenue)
                + _rep(seg.nwc_pct_delta_revenue) * g / (1.0 + g))
    return g <= 0.0 or reinvest <= 0.0


def _terminal(seg, base, term):
    return _rep(term) if term is not None else _rep(base)


def config_checks(segments) -> list[Check]:
    """Checks that need only the segment inputs (also usable before a run)."""
    out = []
    for seg in segments:
        wacc = _rep(seg.wacc)
        g = _rep(seg.terminal_growth_rate)
        if seg.terminal_method == TerminalValueMethod.GORDON_GROWTH:
            spread = wacc - g
            st = FAIL if spread <= 0 else (WARN if spread < MIN_WACC_TVG_SPREAD else OK)
            out.append(Check("4", f"TV-g ≪ WACC · {seg.name}", st,
                             f"TV-g {g:.1%} vs. WACC {wacc:.1%} (Abstand {spread * 100:.1f} pp, "
                             f"Warnung unter {MIN_WACC_TVG_SPREAD * 100:.0f} pp)"))
        cx = _terminal(seg, seg.capex_pct_revenue, seg.capex_pct_revenue_terminal)
        da = _terminal(seg, seg.da_pct_revenue, seg.da_pct_revenue_terminal)
        g_long = g if (seg.terminal_method == TerminalValueMethod.GORDON_GROWTH
                       or seg.revenue_growth_mode == RevenueGrowthMode.FADE) else _rep(seg.revenue_growth)
        if g_long > 0 and cx <= da:
            out.append(Check("7", f"Capex > D&A im Endwert · {seg.name}", WARN,
                             f"Capex {cx:.1%} ≤ D&A {da:.1%} bei g {g_long:.1%}: dauerhaftes Wachstum "
                             f"ohne Nettoinvestition. D&A terminal unter Capex setzen"))
        else:
            out.append(Check("7", f"Capex > D&A im Endwert · {seg.name}", OK,
                             f"Capex {cx:.1%} vs. D&A {da:.1%}"))
    return out


def result_checks(config, results, metadata: dict | None = None,
                  ist_fcff: float | None = None) -> list[Check]:
    """All checks, including those that need simulation results or actuals."""
    out = config_checks(config.segments)
    by_name = {s.name: s for s in config.segments}
    for name, tv in results.segment_tv_ev_ratios.items():
        m = float(np.median(tv))
        out.append(Check("5", f"TV/EV · {name}", WARN if m >= TV_EV_WARN else OK,
                         f"Median {m:.0%} (Warnung ab {TV_EV_WARN:.0%}; Kennzahl ist auf 100 % gekappt)"))
    for name, p in results.segment_prob_value_destruction.items():
        seg = by_name.get(name)
        if p > 0.5 and seg is not None and roic_is_artifact(seg):  # only where a warning would fire
            out.append(Check("6", f"ROIC vs. WACC · {name}", INFO,
                             f"P(ROIC < WACC) {p:.0%} ist ein Formelartefakt: Wachstum oder "
                             f"Reinvestition ≤ 0, Implied ROIC = g × NOPAT / Reinvestition ist dann nicht aussagekräftig"))
        else:
            out.append(Check("6", f"ROIC vs. WACC · {name}", WARN if p > 0.5 else OK,
                             f"P(ROIC < WACC) {p:.0%}"))

    kurs = (metadata or {}).get("kurs_bei_recherche")
    if kurs:
        p50 = float(np.median(results.price_per_share))
        gap = p50 / float(kurs) - 1.0
        out.append(Check("9", "Fair Value vs. Kurs", WARN if abs(gap) > FAIR_VALUE_GAP_WARN else OK,
                         f"P50 {p50:,.2f} vs. Kurs {float(kurs):,.2f} ({gap:+.0%}). "
                         + ("Lücke erklären (Reverse DCF, Annahmen gegen Evidenz)" if abs(gap) > FAIR_VALUE_GAP_WARN else "")))

    model = (sum(year1_fcff(s) * s.ownership_share for s in config.segments)
             - config.corporate_bridge.annual_corporate_costs)
    if ist_fcff:
        dev = model / ist_fcff - 1.0
        st = OK if abs(dev) <= FCFF_TOL_OK else (WARN if abs(dev) <= FCFF_TOL_WARN else FAIL)
        out.append(Check("11", "Jahr-1-FCFF vs. Ist", st,
                         f"Modell {model:,.0f} vs. Ist {ist_fcff:,.0f} ({dev:+.0%})"))
    else:
        out.append(Check("11", "Jahr-1-FCFF vs. Ist", INFO,
                         f"Modell {model:,.0f} (Segmente anteilig − Holdingkosten). Ist-Wert eintragen, um zu vergleichen"))
    return out
