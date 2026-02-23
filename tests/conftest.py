"""
Shared fixtures for the SOTP Monte-Carlo DCF test suite.

All fixtures are deterministic (seeded RNG) so tests are reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    DistributionType,
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    TerminalValueMethod,
)
from application.portfolio_service import AssetInput


# ── RNG ───────────────────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy RNG for reproducible tests."""
    return np.random.default_rng(42)


# ── Distribution configs ─────────────────────────────────────────────────

@pytest.fixture
def fixed_config() -> DistributionConfig:
    return DistributionConfig(dist_type=DistributionType.FIXED, fixed_value=0.05)


@pytest.fixture
def normal_config() -> DistributionConfig:
    return DistributionConfig(
        dist_type=DistributionType.NORMAL, mean=0.05, std=0.01,
    )


@pytest.fixture
def lognormal_config() -> DistributionConfig:
    return DistributionConfig(
        dist_type=DistributionType.LOGNORMAL, ln_mu=0.0, ln_sigma=0.1,
    )


@pytest.fixture
def triangular_config() -> DistributionConfig:
    return DistributionConfig(
        dist_type=DistributionType.TRIANGULAR, low=0.02, mode=0.05, high=0.08,
    )


@pytest.fixture
def uniform_config() -> DistributionConfig:
    return DistributionConfig(
        dist_type=DistributionType.UNIFORM, low=0.03, high=0.07,
    )


@pytest.fixture
def pert_config() -> DistributionConfig:
    return DistributionConfig(
        dist_type=DistributionType.PERT, low=0.02, mode=0.05, high=0.09,
    )


# ── Segment configs ──────────────────────────────────────────────────────

@pytest.fixture
def simple_segment() -> SegmentConfig:
    """Deterministic single segment for fast tests."""
    return SegmentConfig(
        name="TestSeg",
        base_revenue=1_000.0,
        forecast_years=5,
    )


@pytest.fixture
def stochastic_segment() -> SegmentConfig:
    """Segment with stochastic revenue growth and WACC."""
    return SegmentConfig(
        name="StochSeg",
        base_revenue=2_000.0,
        forecast_years=5,
        revenue_growth=DistributionConfig(
            dist_type=DistributionType.NORMAL, mean=0.08, std=0.02,
        ),
        wacc=DistributionConfig(
            dist_type=DistributionType.NORMAL, mean=0.10, std=0.01,
        ),
    )


@pytest.fixture
def fade_segment() -> SegmentConfig:
    """Segment using the fade revenue growth model."""
    return SegmentConfig(
        name="FadeSeg",
        base_revenue=1_500.0,
        forecast_years=7,
        revenue_growth_mode=RevenueGrowthMode.FADE,
        fade_speed=0.5,
        revenue_growth=DistributionConfig(
            dist_type=DistributionType.FIXED, fixed_value=0.15,
        ),
        terminal_growth_rate=DistributionConfig(
            dist_type=DistributionType.FIXED, fixed_value=0.02,
        ),
    )


# ── Simulation configs ───────────────────────────────────────────────────

@pytest.fixture
def minimal_sim_config(simple_segment) -> SimulationConfig:
    """Minimal simulation config with one deterministic segment."""
    return SimulationConfig(
        n_simulations=500,
        random_seed=42,
        segments=[simple_segment],
        corporate_bridge=CorporateBridgeConfig(
            annual_corporate_costs=50.0,
            corporate_cost_discount_rate=0.09,
            net_debt=200.0,
            shares_outstanding=100.0,
        ),
    )


@pytest.fixture
def full_sim_config(stochastic_segment, fade_segment) -> SimulationConfig:
    """Multi-segment config with stochastic bridge for integration tests."""
    return SimulationConfig(
        n_simulations=2_000,
        random_seed=123,
        segments=[stochastic_segment, fade_segment],
        corporate_bridge=CorporateBridgeConfig(
            annual_corporate_costs=40.0,
            corporate_cost_discount_rate=0.09,
            net_debt=500.0,
            shares_outstanding=150.0,
            minority_interests=50.0,
            pension_liabilities=80.0,
            non_operating_assets=120.0,
            associate_investments=60.0,
        ),
        mid_year_convention=True,
    )


# ── Portfolio fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_assets() -> list[AssetInput]:
    """Three deterministic-seeded assets for portfolio tests."""
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(20)
    rng3 = np.random.default_rng(30)
    return [
        AssetInput("Alpha", "Technologie", 100.0, rng1.normal(120, 20, 5_000), 0.0, 1.0),
        AssetInput("Beta", "Energie", 50.0, rng2.normal(55, 10, 5_000), 0.0, 1.0),
        AssetInput("Gamma", "Gesundheit", 80.0, rng3.normal(90, 15, 5_000), 0.0, 1.0),
    ]
