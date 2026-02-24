"""
Application Service Layer – DEPRECATED.

This module is retained for backward compatibility.
Prefer direct usage of ``MonteCarloEngine`` and ``domain.statistics``.
"""
from __future__ import annotations

import numpy as np

from domain.models import SimulationConfig, SimulationResults
from domain.statistics import (
    compute_statistics as _compute_statistics,
    compute_sensitivity as _compute_sensitivity,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine


class SimulationService:
    """Deprecated thin wrapper – use MonteCarloEngine directly."""

    @staticmethod
    def run_simulation(config: SimulationConfig) -> SimulationResults:
        return MonteCarloEngine(config).run()

    @staticmethod
    def compute_sensitivity(results: SimulationResults) -> dict[str, float]:
        return _compute_sensitivity(results.equity_values, results.input_samples)

    @staticmethod
    def compute_statistics(arr: np.ndarray) -> dict[str, float]:
        return _compute_statistics(arr)
