"""
Valuation Quality Metrics – vectorised helpers.
================================================

Pure functions operating on numpy arrays produced by the DCF engine.
No I/O, no side-effects.

Metrics
-------
* **TV / EV ratio** – fraction of enterprise value attributable to the
  terminal value.  High ratios (>70 %) indicate heavy reliance on
  long-term assumptions → fragile valuation.
* **Implied ROIC** – return on invested capital implied by the model's
  margin, reinvestment and growth assumptions.  Benchmark against the
  company's actual ROIC to sanity-check the projection.
* **Reinvestment rate** – share of NOPAT ploughed back into the business.
* **Valuation quality score** – composite 0 – 100 score aggregating
  TV/EV risk, convergence quality, sensitivity concentration and
  outcome dispersion.
"""
from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# TV / EV decomposition
# ═══════════════════════════════════════════════════════════════════════════

def tv_ev_ratio(pv_tv: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Terminal-value share of enterprise value per simulation.

    Returns
    -------
    ratio : (n,)  values in [0, 1] (clamped for safety).
    """
    return np.clip(pv_tv / np.maximum(np.abs(ev), 1e-6), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Implied ROIC
# ═══════════════════════════════════════════════════════════════════════════

def implied_roic(
    ebitda_margin: np.ndarray,
    da_pct_revenue: np.ndarray,
    tax_rate: np.ndarray,
    capex_pct_revenue: np.ndarray,
    nwc_pct_delta_revenue: np.ndarray,
    revenue_growth: np.ndarray,
) -> np.ndarray:
    """Steady-state implied ROIC from DCF model assumptions.

    In a steady state the company grows at *g* and reinvests a fraction
    of NOPAT.  The implied ROIC is:

    .. math::

        \\text{NOPAT margin} = (\\text{EBITDA%} - \\text{D\\&A%}) \\times (1 - t)

        \\text{reinvest}_\\text{margin} = \\text{CAPEX%} - \\text{D\\&A%}
            + \\text{NWC%} \\times \\frac{g}{1 + g}

        \\text{ROIC} = \\frac{\\text{NOPAT margin}}{\\text{reinvest margin}}

    Parameters
    ----------
    All inputs are (n,) arrays of sampled values.

    Returns
    -------
    roic : (n,)  – implied ROIC (clamped to [-2, +5] for sanity).
    """
    nopat_margin = (ebitda_margin - da_pct_revenue) * (1.0 - tax_rate)
    # Net reinvestment as fraction of revenue
    # NWC change per unit revenue ≈ nwc% × g/(1+g) (steady-state approx)
    g_safe = np.clip(revenue_growth, -0.5, 0.99)
    nwc_reinvest = nwc_pct_delta_revenue * g_safe / (1.0 + g_safe)
    reinvest_margin = capex_pct_revenue - da_pct_revenue + nwc_reinvest

    roic = nopat_margin / np.maximum(np.abs(reinvest_margin), 1e-6)
    # Sign correction: if reinvest_margin is negative (net cash release),
    # ROIC is mathematically negative but economically meaningless –
    # clamp for display purposes.
    return np.clip(roic, -2.0, 5.0)


def reinvestment_rate(
    capex_pct_revenue: np.ndarray,
    da_pct_revenue: np.ndarray,
    nwc_pct_delta_revenue: np.ndarray,
    revenue_growth: np.ndarray,
    ebitda_margin: np.ndarray,
    tax_rate: np.ndarray,
) -> np.ndarray:
    """Fraction of NOPAT reinvested (net CAPEX + ΔNWC) / NOPAT.

    Returns
    -------
    rate : (n,)  – clamped to [-1, 2].
    """
    nopat_margin = (ebitda_margin - da_pct_revenue) * (1.0 - tax_rate)
    g_safe = np.clip(revenue_growth, -0.5, 0.99)
    nwc_reinvest = nwc_pct_delta_revenue * g_safe / (1.0 + g_safe)
    net_reinvest = capex_pct_revenue - da_pct_revenue + nwc_reinvest
    return np.clip(
        net_reinvest / np.maximum(np.abs(nopat_margin), 1e-6),
        -1.0,
        2.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Valuation quality score  (composite 0 – 100)
# ═══════════════════════════════════════════════════════════════════════════

def _score_tv_ev(mean_tv_ev: float) -> float:
    """0–25 points.  ≤40 % TV/EV → full score; ≥90 % → zero."""
    return max(0.0, min(25.0, 25.0 * (0.90 - mean_tv_ev) / 0.50))


def _score_convergence(ci_width_pct: float) -> float:
    """0–25 points.  <0.5 % → full; >5 % → zero."""
    return max(0.0, min(25.0, 25.0 * (5.0 - ci_width_pct) / 4.5))


def _score_sensitivity(correlations: dict[str, float]) -> float:
    """0–25 points.  Herfindahl index of squared correlations.

    Low concentration (many equally important drivers) → high score.
    """
    if not correlations:
        return 25.0  # deterministic model – no sensitivity risk
    sq = np.array([v ** 2 for v in correlations.values()])
    total = sq.sum()
    if total < 1e-12:
        return 25.0
    hhi = float(np.sum((sq / total) ** 2))
    # HHI ∈ [1/N, 1].  1/N → perfect diversification → 25 pts.
    return max(0.0, min(25.0, 25.0 * (1.0 - hhi)))


def _score_dispersion(cv: float) -> float:
    """0–25 points.  CV (CoeffVar) = σ/|μ|.  <0.1 → 25; >1.0 → 0."""
    return max(0.0, min(25.0, 25.0 * (1.0 - cv) / 0.9))


def valuation_quality_score(
    mean_tv_ev: float,
    ci_width_pct: float,
    correlations: dict[str, float],
    equity_mean: float,
    equity_std: float,
) -> dict[str, float]:
    """Composite quality score with sub-component breakdown.

    Parameters
    ----------
    mean_tv_ev : float
        Average TV/EV ratio across simulations (0–1).
    ci_width_pct : float
        95 % CI width as percentage of mean from convergence diagnostic.
    correlations : dict
        Spearman correlations from sensitivity analysis.
    equity_mean, equity_std : float
        Mean and std of equity value distribution.

    Returns
    -------
    dict with keys: ``total``, ``tv_ev``, ``convergence``,
    ``sensitivity``, ``dispersion``.  All floats.
    """
    cv = equity_std / max(abs(equity_mean), 1e-6)

    s_tv = _score_tv_ev(mean_tv_ev)
    s_conv = _score_convergence(ci_width_pct)
    s_sens = _score_sensitivity(correlations)
    s_disp = _score_dispersion(cv)

    return {
        "total": s_tv + s_conv + s_sens + s_disp,
        "tv_ev": s_tv,
        "convergence": s_conv,
        "sensitivity": s_sens,
        "dispersion": s_disp,
    }
