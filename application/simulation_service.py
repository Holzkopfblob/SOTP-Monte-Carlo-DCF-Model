"""
Application Service Layer.

Provides a high-level API that the presentation layer consumes.
Orchestrates the Monte-Carlo engine and post-processing (sensitivity
analysis, descriptive statistics).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats as sp_stats

from domain.models import SimulationConfig, SimulationResults
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
    def compute_sensitivity(results: SimulationResults) -> Dict[str, float]:
        """
        Spearman rank-correlation of each stochastic input with equity value.

        Returns a dict sorted by **absolute** correlation (descending).
        Constant inputs (σ ≈ 0) are silently excluded.
        """
        target = results.equity_values
        correlations: Dict[str, float] = {}

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
    # Descriptive statistics helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_statistics(arr: np.ndarray) -> Dict[str, float]:
        """Full descriptive statistics for a 1-D array of outcomes."""
        return {
            "Mittelwert":  float(np.mean(arr)),
            "Median":      float(np.median(arr)),
            "Std.-Abw.":   float(np.std(arr)),
            "P5 (5%)":     float(np.percentile(arr, 5)),
            "P25 (25%)":   float(np.percentile(arr, 25)),
            "P75 (75%)":   float(np.percentile(arr, 75)),
            "P95 (95%)":   float(np.percentile(arr, 95)),
            "Min":         float(np.min(arr)),
            "Max":         float(np.max(arr)),
        }
