"""
Domain Models for the SOTP Monte-Carlo DCF Simulation.

Contains all value objects and data transfer structures used
across the application layers.  These are pure data – no I/O,
no side-effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DistributionType(str, Enum):
    """Supported probability distribution types."""
    FIXED       = "Fest (Deterministisch)"
    NORMAL      = "Normalverteilung"
    LOGNORMAL   = "Lognormalverteilung"
    TRIANGULAR  = "Dreiecksverteilung"
    UNIFORM     = "Gleichverteilung"
    PERT        = "PERT-Verteilung"


class TerminalValueMethod(str, Enum):
    """Terminal value calculation methods."""
    GORDON_GROWTH = "Gordon Growth Model (Ewige Rente)"
    EXIT_MULTIPLE = "Exit-Multiple-Ansatz"


# ---------------------------------------------------------------------------
# Distribution configuration
# ---------------------------------------------------------------------------

@dataclass
class DistributionConfig:
    """
    Configuration for a single stochastic or deterministic parameter.

    Stores parameters for *all* distribution types.  Only those matching
    ``dist_type`` are used at sampling time.
    """
    dist_type: DistributionType = DistributionType.FIXED

    # Fixed
    fixed_value: float = 0.0

    # Normal  (μ, σ)
    mean: float = 0.0
    std:  float = 0.01

    # LogNormal  (underlying normal μ, σ)
    ln_mu:    float = 0.0
    ln_sigma: float = 0.1

    # Triangular / PERT  (low, mode, high)
    low:  float = 0.0
    mode: float = 0.0
    high: float = 0.1

    # Uniform  (low, high)  – reuse ``low`` / ``high`` would be ambiguous,
    # so we keep separate fields for clarity when both Triangular *and*
    # Uniform are options.
    # NOTE: For Triangular/PERT we use low/mode/high above.
    #       For Uniform we also use low/high above (mode is ignored).


# ---------------------------------------------------------------------------
# Segment configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """Configuration for one business segment (FCFF-based DCF)."""
    name: str = "Segment"
    base_revenue: float = 1_000.0       # Mio.
    forecast_years: int = 5

    # Value drivers – each can be deterministic or stochastic
    revenue_growth:        DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.05))
    ebitda_margin:         DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.20))
    da_pct_revenue:        DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.03))
    tax_rate:              DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.25))
    capex_pct_revenue:     DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.05))
    nwc_pct_delta_revenue: DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.10))
    wacc:                  DistributionConfig = field(default_factory=lambda: DistributionConfig(fixed_value=0.09))

    # Terminal value
    terminal_method:      TerminalValueMethod = TerminalValueMethod.GORDON_GROWTH
    terminal_growth_rate: DistributionConfig   = field(default_factory=lambda: DistributionConfig(fixed_value=0.02))
    exit_multiple:        DistributionConfig   = field(default_factory=lambda: DistributionConfig(fixed_value=10.0))


# ---------------------------------------------------------------------------
# Corporate bridge
# ---------------------------------------------------------------------------

@dataclass
class CorporateBridgeConfig:
    """Corporate-level adjustments for the equity bridge."""
    annual_corporate_costs:       float = 50.0    # Mio. per annum
    corporate_cost_discount_rate: float = 0.09    # for perpetuity PV
    net_debt:                     float = 500.0   # Mio.
    shares_outstanding:           float = 100.0   # Mio. shares


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Complete configuration for one simulation run."""
    n_simulations: int             = 10_000
    random_seed:   int             = 42
    segments:      List[SegmentConfig]     = field(default_factory=list)
    corporate_bridge: CorporateBridgeConfig = field(default_factory=CorporateBridgeConfig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SimulationResults:
    """Holds all results of a completed Monte Carlo simulation."""
    # Core output vectors  (n_sim,)
    equity_values:      np.ndarray
    total_ev:           np.ndarray
    segment_evs:        Dict[str, np.ndarray]       # segment name → (n_sim,)
    pv_corporate_costs: np.ndarray

    # Input samples for sensitivity analysis
    input_samples: Dict[str, np.ndarray]            # param label → (n_sim,)

    # Base-case (expectation) values for waterfall chart
    base_segment_evs:       Dict[str, float] = field(default_factory=dict)
    base_corporate_costs_pv: float = 0.0
    base_net_debt:           float = 0.0
    base_equity_value:       float = 0.0

    # Derived
    price_per_share: np.ndarray = field(default_factory=lambda: np.array([]))
    n_simulations:   int = 0
