"""
Parameter Fade-Curve Builder (Phase 3).

Provides a reusable function to compute the exponential-decay fade
path from an initial to a terminal value over T years.  Used by
the Monte-Carlo engine for revenue growth *and* all other FCFF
parameters (EBITDA margin, D&A%, tax, CAPEX%, NWC%).
"""
from __future__ import annotations

import numpy as np


def build_fade_curve(
    initial: np.ndarray,      # (n,) – sampled initial values
    terminal: np.ndarray,     # (n,) – sampled terminal values
    forecast_years: int,
    fade_speed: float,
) -> np.ndarray:
    """Create an exponentially decaying parameter path.

    .. math::

        p_t = p_{\\text{terminal}} + (p_{\\text{initial}} - p_{\\text{terminal}})
              \\cdot e^{-\\lambda \\, t}

    Parameters
    ----------
    initial : (n,) array
        Starting values (year-1).
    terminal : (n,) array
        Long-run equilibrium values.
    forecast_years : int *T*
        Number of explicit forecast years.
    fade_speed : float *λ*
        Exponential decay rate (higher → faster convergence).

    Returns
    -------
    curve : (n, T) array
        Time-varying parameter path for each simulation run.
    """
    years = np.arange(1, forecast_years + 1, dtype=np.float64)  # [1 … T]
    decay = np.exp(-fade_speed * years)                          # (T,)

    p_init = initial[:, None]                                    # (n, 1)
    p_term = terminal[:, None]                                   # (n, 1)

    return p_term + (p_init - p_term) * decay[None, :]           # (n, T)
