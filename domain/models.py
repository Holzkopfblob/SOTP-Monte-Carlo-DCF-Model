"""
Domain Models for the SOTP Monte-Carlo DCF Simulation.

Contains all value objects and data transfer structures used
across the application layers.  These are pure data – no I/O,
no side-effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


class RevenueGrowthMode(str, Enum):
    """Revenue growth modelling approach."""
    CONSTANT = "Konstant (g über alle Jahre gleich)"
    FADE     = "Fade-Modell (g konvergiert zum Terminal-Wachstum)"


class SamplingMethod(str, Enum):
    """Variance-reduction / quasi-random sampling strategies."""
    PSEUDO_RANDOM = "Pseudo-Random (Standard)"
    ANTITHETIC    = "Antithetic Variates"
    SOBOL         = "Quasi-MC (Sobol)"


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

    def representative_value(self) -> float:
        """Return a single representative (central) value for previews.

        This is NOT a statistical mean – it is a best-effort point
        estimate used for deterministic fade-curve previews and similar
        UI helpers where a single number is needed.
        """
        dt = self.dist_type
        if dt == DistributionType.FIXED:
            return self.fixed_value
        if dt == DistributionType.NORMAL:
            return self.mean
        if dt == DistributionType.LOGNORMAL:
            return self.ln_mu
        if dt in (DistributionType.TRIANGULAR, DistributionType.PERT):
            return self.mode
        if dt == DistributionType.UNIFORM:
            return (self.low + self.high) / 2.0
        return self.fixed_value  # fallback


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

    # Revenue growth mode (constant vs. fade)
    revenue_growth_mode: RevenueGrowthMode = RevenueGrowthMode.CONSTANT
    fade_speed: float = 0.5             # λ – exponential decay speed (higher = faster fade)

    # ── Phase 3: Parameter fade (terminal targets) ────────────────────
    # When not None, the parameter fades from its initial distribution
    # to the terminal distribution using the same exponential decay:
    #   p_t = p_terminal + (p_initial - p_terminal) * exp(-λ * t)
    ebitda_margin_terminal:         DistributionConfig | None = None
    da_pct_revenue_terminal:        DistributionConfig | None = None
    tax_rate_terminal:              DistributionConfig | None = None
    capex_pct_revenue_terminal:     DistributionConfig | None = None
    nwc_pct_delta_revenue_terminal: DistributionConfig | None = None

    # ── Phase 4: Intra-segment parameter correlation ──────────────────
    # Optional 7×7 correlation matrix for the value drivers within this
    # segment.  Parameter order:
    #   [revenue_growth, ebitda_margin, da_pct, tax_rate,
    #    capex_pct, nwc_pct, wacc]
    # When not None, a Gaussian copula is used so that each parameter
    # retains its marginal distribution but draws become dependent.
    intra_param_correlation: list[list[float]] | None = None


# ── Default intra-segment correlation matrix ──────────────────────────────
# Sensible empirical defaults reflecting typical accounting identities:
#   revenue_growth ↔ ebitda_margin: moderate positive (operating leverage)
#   ebitda_margin ↔ capex: moderate positive (margin expansion needs investment)
#   capex ↔ da_pct: strong positive (more capex → more depreciation)
#   revenue_growth ↔ nwc: moderate positive (growth ties working capital)
#   wacc: mildly correlated with margins (risk ↔ profitability)

INTRA_PARAM_LABELS = [
    "Umsatzwachstum", "EBITDA-Marge", "D&A (% Umsatz)",
    "Steuersatz", "CAPEX (% Umsatz)", "NWC (% ΔUmsatz)", "WACC",
]

#                              rg    ebitda  da    tax   capex  nwc   wacc
DEFAULT_INTRA_PARAM_CORR = [
    [ 1.00,  0.30,  0.10, 0.00,  0.15,  0.35, -0.10],  # revenue_growth
    [ 0.30,  1.00,  0.20, 0.00,  0.25,  0.10, -0.20],  # ebitda_margin
    [ 0.10,  0.20,  1.00, 0.00,  0.70,  0.05,  0.00],  # da_pct
    [ 0.00,  0.00,  0.00, 1.00,  0.00,  0.00,  0.10],  # tax_rate
    [ 0.15,  0.25,  0.70, 0.00,  1.00,  0.10,  0.05],  # capex_pct
    [ 0.35,  0.10,  0.05, 0.00,  0.10,  1.00,  0.00],  # nwc_pct
    [-0.10, -0.20,  0.00, 0.10,  0.05,  0.00,  1.00],  # wacc
]


# ---------------------------------------------------------------------------
# Corporate bridge
# ---------------------------------------------------------------------------

@dataclass
class CorporateBridgeConfig:
    """Corporate-level adjustments for the equity bridge.

    All four parameters can be modelled stochastically.
    The scalar defaults are kept for backward compatibility;
    when ``stochastic_*`` configs are not None they override
    the scalar value during simulation.
    """
    annual_corporate_costs:       float = 50.0    # Mio. per annum
    corporate_cost_discount_rate: float = 0.09    # for perpetuity PV
    net_debt:                     float = 500.0   # Mio.
    shares_outstanding:           float = 100.0   # Mio. shares

    # Extended bridge items
    minority_interests:      float = 0.0     # Mio. (ownership by third parties in subsidiaries)
    pension_liabilities:     float = 0.0     # Mio. (unfunded pension obligations)
    non_operating_assets:    float = 0.0     # Mio. (excess cash, real estate, investments)
    associate_investments:   float = 0.0     # Mio. (equity-method investments)

    # Stochastic overrides (None = use scalar above)
    stochastic_corporate_costs: DistributionConfig | None = None
    stochastic_corporate_cost_discount_rate: DistributionConfig | None = None
    stochastic_net_debt:        DistributionConfig | None = None
    stochastic_shares:          DistributionConfig | None = None
    stochastic_minority_interests:    DistributionConfig | None = None
    stochastic_pension_liabilities:   DistributionConfig | None = None
    stochastic_non_operating_assets:  DistributionConfig | None = None
    stochastic_associate_investments: DistributionConfig | None = None


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Complete configuration for one simulation run."""
    n_simulations: int             = 10_000
    random_seed:   int             = 42
    segments:      list[SegmentConfig]     = field(default_factory=list)
    corporate_bridge: CorporateBridgeConfig = field(default_factory=CorporateBridgeConfig)
    mid_year_convention: bool      = True     # discount FCFFs at t−0.5 (standard practice)

    # ── Sampling method ───────────────────────────────────────────────
    sampling_method: SamplingMethod = SamplingMethod.PSEUDO_RANDOM

    # ── Phase 3: Cross-segment correlation ────────────────────────────
    # Correlation matrix (n_seg × n_seg) stored as list-of-lists.
    # When not None, a Gaussian copula correlates all stochastic draws
    # between segments.  Diagonal must be 1.0; off-diag in [−1, 1].
    segment_correlation: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# Segment EV decomposition (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class SegmentEVDetail:
    """Decomposed segment enterprise-value result.

    Returned by ``compute_segment_ev`` so that the engine can extract
    PV(FCFF) and PV(TV) separately for quality-metric computation.
    """
    ev:      np.ndarray   # (n,) = pv_fcff + pv_tv
    pv_fcff: np.ndarray   # (n,)
    pv_tv:   np.ndarray   # (n,)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SimulationResults:
    """Holds all results of a completed Monte Carlo simulation."""
    # Core output vectors  (n_sim,)
    equity_values:      np.ndarray
    total_ev:           np.ndarray
    segment_evs:        dict[str, np.ndarray]       # segment name → (n_sim,)
    pv_corporate_costs: np.ndarray

    # Input samples for sensitivity analysis
    input_samples: dict[str, np.ndarray]            # param label → (n_sim,)

    # Base-case (expectation) values for waterfall chart
    base_segment_evs:       dict[str, float] = field(default_factory=dict)
    base_corporate_costs_pv: float = 0.0
    base_net_debt:           float = 0.0
    base_equity_value:       float = 0.0

    # Extended bridge base-case values
    base_minority_interests:    float = 0.0
    base_pension_liabilities:   float = 0.0
    base_non_operating_assets:  float = 0.0
    base_associate_investments: float = 0.0

    # Derived
    price_per_share: np.ndarray = field(default_factory=lambda: np.array([]))
    n_simulations:   int = 0

    # Convergence diagnostics – cumulative running mean at checkpoint indices
    convergence_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    convergence_means:   np.ndarray = field(default_factory=lambda: np.array([]))
    convergence_ci_low:  np.ndarray = field(default_factory=lambda: np.array([]))
    convergence_ci_high: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Phase 2: Core Insights ────────────────────────────────────────
    # Per-segment TV/EV ratio distributions  (segment name → (n_sim,))
    segment_tv_ev_ratios: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-segment implied ROIC distributions  (segment name → (n_sim,))
    segment_implied_roic: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-segment reinvestment rates  (segment name → (n_sim,))
    segment_reinvest_rates: dict[str, np.ndarray] = field(default_factory=dict)
    # Composite quality score dict (total, tv_ev, convergence, sensitivity, dispersion)
    quality_score: dict[str, float] = field(default_factory=dict)
