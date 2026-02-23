"""
Application Service Layer.

Provides a high-level API that the presentation layer consumes.
Orchestrates the Monte-Carlo engine and post-processing (sensitivity
analysis, descriptive statistics).
"""
from __future__ import annotations


import numpy as np
from scipy import stats as sp_stats

from domain.models import SimulationConfig, SimulationResults
from domain.statistics import compute_statistics as _compute_statistics
from infrastructure.monte_carlo_engine import MonteCarloEngine


class SimulationService:
    """Stateless service – all methods are static for convenience."""

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def run_simulation(config: SimulationConfig) -> SimulationResults:
        """Run the full SOTP Monte-Carlo DCF and return results."""
        engine = MonteCarloEngine(config)
        return engine.run()

    # ------------------------------------------------------------------
    # Sensitivity analysis (Tornado chart data)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sensitivity(results: SimulationResults) -> dict[str, float]:
        """
        Spearman rank-correlation of each stochastic input with equity value.

        Returns a dict sorted by **absolute** correlation (descending).
        Constant inputs (σ ≈ 0) are silently excluded.
        """
        target = results.equity_values
        correlations: dict[str, float] = {}

        for name, samples in results.input_samples.items():
            if np.std(samples) < 1e-12:
                continue  # skip deterministic (fixed) parameters
            corr, _ = sp_stats.spearmanr(samples, target)
            if np.isfinite(corr):
                correlations[name] = float(corr)

        return dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    # ------------------------------------------------------------------
    # Descriptive statistics helper  (delegates to domain.statistics)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_statistics(arr: np.ndarray) -> dict[str, float]:
        """Full descriptive statistics for a 1-D array of outcomes.

        Backward-compatible delegate – canonical implementation lives
        in :func:`domain.statistics.compute_statistics`.
        """
        return _compute_statistics(arr)
