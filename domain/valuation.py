"""
Core Financial Valuation Logic – Vectorised FCFF-based DCF.

Every function operates on **numpy arrays** so that the entire
Monte-Carlo simulation runs without Python-level loops.
"""
from __future__ import annotations

import numpy as np

from domain.models import RevenueGrowthMode, TerminalValueMethod


# ═══════════════════════════════════════════════════════════════════════════
# FCFF schedule
# ═══════════════════════════════════════════════════════════════════════════

def compute_fcff_vectors(
    base_revenue: float,
    forecast_years: int,
    revenue_growth: np.ndarray,          # (n,)
    ebitda_margin: np.ndarray,           # (n,)
    da_pct_revenue: np.ndarray,          # (n,)
    tax_rate: np.ndarray,                # (n,)
    capex_pct_revenue: np.ndarray,       # (n,)
    nwc_pct_delta_revenue: np.ndarray,   # (n,)
    *,
    growth_mode: RevenueGrowthMode = RevenueGrowthMode.CONSTANT,
    terminal_growth: np.ndarray | None = None,
    fade_speed: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full FCFF schedule for a single segment.

    Parameters
    ----------
    base_revenue : float
        Year-0 revenue (the starting point).
    forecast_years : int
        Number of explicit forecast years (T).
    revenue_growth … nwc_pct_delta_revenue : (n,) arrays
        One sampled value per simulation.
    growth_mode : RevenueGrowthMode
        CONSTANT – same g every year (original behaviour).
        FADE     – g decays exponentially from initial to terminal growth:
                   g_t = g_terminal + (g_initial - g_terminal) * exp(-λ * t)
    terminal_growth : (n,) array, optional
        Terminal growth rate – used as the long-run target in FADE mode.
    fade_speed : float
        λ parameter controlling how fast g converges (higher = faster).

    Returns
    -------
    fcff    : (n, T) – Free Cash Flow to Firm per year
    ebitda  : (n, T) – EBITDA per year
    revenue : (n, T) – Revenue per year
    """
    n = revenue_growth.shape[0]
    years = np.arange(1, forecast_years + 1, dtype=np.float64)  # [1 … T]

    if growth_mode == RevenueGrowthMode.FADE and terminal_growth is not None:
        # Fade model: g_t = g_term + (g_init - g_term) * exp(-λ * t)
        # revenue_growth = initial growth (g_0), terminal_growth = long-run g
        g_init = revenue_growth[:, None]                     # (n, 1)
        g_term = terminal_growth[:, None]                    # (n, 1)
        decay = np.exp(-fade_speed * years[None, :])         # (1, T)
        g_t = g_term + (g_init - g_term) * decay             # (n, T)

        # Build revenue year by year using compounding with varying g
        revenue = np.empty((n, forecast_years), dtype=np.float64)
        revenue[:, 0] = base_revenue * (1.0 + g_t[:, 0])
        for t in range(1, forecast_years):
            revenue[:, t] = revenue[:, t - 1] * (1.0 + g_t[:, t])
    else:
        # Original constant-growth model
        revenue = base_revenue * np.power(
            (1.0 + revenue_growth)[:, None],
            years[None, :],
        )

    ebitda = revenue * ebitda_margin[:, None]
    da     = revenue * da_pct_revenue[:, None]
    ebit   = ebitda - da

    # Taxes only on positive EBIT
    taxes  = np.maximum(ebit, 0.0) * tax_rate[:, None]
    nopat  = ebit - taxes

    capex  = revenue * capex_pct_revenue[:, None]

    # Change in net working capital
    prev_rev = np.empty_like(revenue)
    prev_rev[:, 0]  = base_revenue
    prev_rev[:, 1:] = revenue[:, :-1]
    delta_nwc = (revenue - prev_rev) * nwc_pct_delta_revenue[:, None]

    # FCFF = NOPAT + D&A − CAPEX − ΔNWC
    fcff = nopat + da - capex - delta_nwc

    return fcff, ebitda, revenue


# ═══════════════════════════════════════════════════════════════════════════
# Terminal value
# ═══════════════════════════════════════════════════════════════════════════

def compute_terminal_value(
    method: TerminalValueMethod,
    fcff_last: np.ndarray,      # (n,)
    ebitda_last: np.ndarray,    # (n,)
    wacc: np.ndarray,           # (n,)
    terminal_growth: np.ndarray,  # (n,)
    exit_multiple: np.ndarray,    # (n,)
) -> np.ndarray:
    """
    Terminal value vector (n,).

    Gordon Growth:  TV = FCFF_T × (1 + g) / (WACC − g)
    Exit Multiple:  TV = EBITDA_T × Multiple
    """
    if method == TerminalValueMethod.GORDON_GROWTH:
        denom = np.maximum(wacc - terminal_growth, 1e-4)
        return fcff_last * (1.0 + terminal_growth) / denom
    else:
        return ebitda_last * exit_multiple


# ═══════════════════════════════════════════════════════════════════════════
# Segment enterprise value
# ═══════════════════════════════════════════════════════════════════════════

def compute_segment_ev(
    base_revenue: float,
    forecast_years: int,
    revenue_growth: np.ndarray,
    ebitda_margin: np.ndarray,
    da_pct_revenue: np.ndarray,
    tax_rate: np.ndarray,
    capex_pct_revenue: np.ndarray,
    nwc_pct_delta_revenue: np.ndarray,
    wacc: np.ndarray,
    terminal_method: TerminalValueMethod,
    terminal_growth: np.ndarray,
    exit_multiple: np.ndarray,
    *,
    growth_mode: RevenueGrowthMode = RevenueGrowthMode.CONSTANT,
    fade_speed: float = 0.5,
) -> np.ndarray:
    """
    Full DCF valuation for a single segment.

    Returns
    -------
    ev : (n,) – Enterprise value per simulation.
    """
    fcff, ebitda, _ = compute_fcff_vectors(
        base_revenue, forecast_years,
        revenue_growth, ebitda_margin, da_pct_revenue,
        tax_rate, capex_pct_revenue, nwc_pct_delta_revenue,
        growth_mode=growth_mode,
        terminal_growth=terminal_growth,
        fade_speed=fade_speed,
    )

    years = np.arange(1, forecast_years + 1, dtype=np.float64)
    discount = 1.0 / np.power(
        (1.0 + wacc)[:, None],
        years[None, :],
    )

    pv_fcff = np.sum(fcff * discount, axis=1)           # (n,)

    tv = compute_terminal_value(
        terminal_method,
        fcff[:, -1],
        ebitda[:, -1],
        wacc,
        terminal_growth,
        exit_multiple,
    )
    pv_tv = tv / np.power(1.0 + wacc, float(forecast_years))   # (n,)

    return pv_fcff + pv_tv


# ═══════════════════════════════════════════════════════════════════════════
# Corporate costs present value  (perpetuity approach)
# ═══════════════════════════════════════════════════════════════════════════

def compute_corporate_costs_pv(
    annual_costs: np.ndarray | float,
    discount_rate: np.ndarray,   # (n,)
) -> np.ndarray:
    """PV of perpetual corporate holding costs:  annual_costs / r.

    ``annual_costs`` may be a scalar (backward-compatible) or an (n,) array
    when modelled stochastically.
    """
    safe_rate = np.maximum(discount_rate, 1e-4)
    return np.asarray(annual_costs) / safe_rate
