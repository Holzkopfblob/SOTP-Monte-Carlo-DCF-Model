"""
Tests for domain.distributions – sampling & factory.

Covers all 6 distribution types: Fixed, Normal, LogNormal, Triangular,
Uniform, PERT.  Each test checks shape, determinism (seeded), and
statistical properties (mean within tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.distributions import (
    BaseDistribution,
    FixedDistribution,
    LogNormalDistribution,
    NormalDistribution,
    PERTDistribution,
    TriangularDistribution,
    UniformDistribution,
    create_distribution,
)
from domain.models import DistributionConfig, DistributionType


N = 50_000  # large enough for statistical checks


class TestFixedDistribution:
    def test_returns_constant_array(self, rng):
        d = FixedDistribution(0.42)
        s = d.sample(100, rng)
        assert s.shape == (100,)
        assert np.all(s == 0.42)

    def test_describe(self):
        assert "0.4200" in FixedDistribution(0.42).describe()


class TestNormalDistribution:
    def test_shape_and_mean(self, rng):
        d = NormalDistribution(mean=5.0, std=1.0)
        s = d.sample(N, rng)
        assert s.shape == (N,)
        assert abs(np.mean(s) - 5.0) < 0.05

    def test_std_matches(self, rng):
        d = NormalDistribution(mean=0.0, std=2.0)
        s = d.sample(N, rng)
        assert abs(np.std(s) - 2.0) < 0.05

    def test_zero_std_clamped(self, rng):
        d = NormalDistribution(mean=1.0, std=0.0)
        s = d.sample(10, rng)
        # should not crash; std is clamped to 1e-12
        assert s.shape == (10,)

    def test_deterministic_with_seed(self):
        d = NormalDistribution(0.0, 1.0)
        s1 = d.sample(100, np.random.default_rng(99))
        s2 = d.sample(100, np.random.default_rng(99))
        np.testing.assert_array_equal(s1, s2)


class TestLogNormalDistribution:
    def test_always_positive(self, rng):
        d = LogNormalDistribution(mu=0.0, sigma=0.5)
        s = d.sample(N, rng)
        assert np.all(s > 0)

    def test_shape(self, rng):
        d = LogNormalDistribution(0.0, 0.1)
        assert d.sample(200, rng).shape == (200,)


class TestTriangularDistribution:
    def test_within_bounds(self, rng):
        d = TriangularDistribution(low=1.0, mode=3.0, high=5.0)
        s = d.sample(N, rng)
        assert np.all(s >= 1.0)
        assert np.all(s <= 5.0)

    def test_mode_clamped(self, rng):
        d = TriangularDistribution(low=2.0, mode=1.0, high=4.0)
        # mode < low should be clamped
        assert d.mode >= d.low

    def test_degenerate_range(self, rng):
        d = TriangularDistribution(low=3.0, mode=3.0, high=3.0)
        s = d.sample(10, rng)
        assert s.shape == (10,)


class TestUniformDistribution:
    def test_within_bounds(self, rng):
        d = UniformDistribution(low=2.0, high=8.0)
        s = d.sample(N, rng)
        assert np.all(s >= 2.0)
        assert np.all(s <= 8.0)

    def test_mean_approx(self, rng):
        d = UniformDistribution(low=0.0, high=10.0)
        s = d.sample(N, rng)
        assert abs(np.mean(s) - 5.0) < 0.1


class TestPERTDistribution:
    def test_within_bounds(self, rng):
        d = PERTDistribution(low=1.0, mode=4.0, high=7.0)
        s = d.sample(N, rng)
        assert np.all(s >= 1.0)
        assert np.all(s <= 7.0)

    def test_mode_weight(self, rng):
        """PERT should concentrate more mass near the mode than Triangular."""
        pert = PERTDistribution(low=0.0, mode=5.0, high=10.0)
        tri = TriangularDistribution(low=0.0, mode=5.0, high=10.0)
        s_pert = pert.sample(N, rng)
        s_tri = tri.sample(N, np.random.default_rng(42))  # fresh rng
        # PERT should have smaller std (tighter around mode)
        assert np.std(s_pert) < np.std(s_tri)


class TestFactory:
    def test_fixed(self, fixed_config, rng):
        d = create_distribution(fixed_config)
        assert isinstance(d, FixedDistribution)
        np.testing.assert_array_equal(d.sample(5, rng), np.full(5, 0.05))

    def test_normal(self, normal_config, rng):
        d = create_distribution(normal_config)
        assert isinstance(d, NormalDistribution)
        assert d.sample(10, rng).shape == (10,)

    def test_lognormal(self, lognormal_config, rng):
        d = create_distribution(lognormal_config)
        assert isinstance(d, LogNormalDistribution)

    def test_triangular(self, triangular_config, rng):
        d = create_distribution(triangular_config)
        assert isinstance(d, TriangularDistribution)

    def test_uniform(self, uniform_config, rng):
        d = create_distribution(uniform_config)
        assert isinstance(d, UniformDistribution)

    def test_pert(self, pert_config, rng):
        d = create_distribution(pert_config)
        assert isinstance(d, PERTDistribution)

    def test_unknown_type_raises(self):
        cfg = DistributionConfig()
        cfg.dist_type = "INVALID"
        with pytest.raises(ValueError, match="Unbekannter"):
            create_distribution(cfg)
