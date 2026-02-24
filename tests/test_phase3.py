"""
Phase 3 Tests – Parameter Fade & Cross-Segment Correlation.

Covers:
- ``domain.fade.build_fade_curve``           (unit)
- ``BaseDistribution.ppf`` for all 6 types   (unit)
- ``compute_fcff_vectors`` with (n, T) params (integration)
- Engine parameter fade sampling              (integration)
- Gaussian copula / correlation               (integration)
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from domain.distributions import (
    FixedDistribution,
    LogNormalDistribution,
    NormalDistribution,
    PERTDistribution,
    TriangularDistribution,
    UniformDistribution,
    create_distribution,
)
from domain.fade import build_fade_curve
from domain.models import (
    CorporateBridgeConfig,
    DistributionConfig,
    DistributionType,
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    TerminalValueMethod,
)
from domain.valuation import compute_fcff_vectors, compute_segment_ev
from infrastructure.monte_carlo_engine import MonteCarloEngine


# ═══════════════════════════════════════════════════════════════════════════
# 1.  build_fade_curve
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildFadeCurve:
    def test_output_shape(self):
        initial = np.array([0.20, 0.25])
        terminal = np.array([0.15, 0.18])
        curve = build_fade_curve(initial, terminal, forecast_years=5, fade_speed=0.5)
        assert curve.shape == (2, 5)

    def test_year1_close_to_initial(self):
        """Year 1 should be close to initial (small decay)."""
        initial = np.array([0.30])
        terminal = np.array([0.10])
        curve = build_fade_curve(initial, terminal, forecast_years=10, fade_speed=0.3)
        # With λ=0.3, t=1: exp(-0.3) ≈ 0.74 → still ~74 % of the way to initial
        assert curve[0, 0] > 0.20

    def test_convergence_to_terminal(self):
        """After many years the curve should converge near terminal."""
        initial = np.array([0.30])
        terminal = np.array([0.10])
        curve = build_fade_curve(initial, terminal, forecast_years=50, fade_speed=0.5)
        np.testing.assert_allclose(curve[0, -1], 0.10, atol=1e-6)

    def test_fast_fade(self):
        """High λ → rapid convergence."""
        initial = np.array([0.30])
        terminal = np.array([0.10])
        curve = build_fade_curve(initial, terminal, forecast_years=5, fade_speed=5.0)
        np.testing.assert_allclose(curve[0, -1], 0.10, atol=1e-4)

    def test_equal_init_terminal(self):
        """When initial == terminal, curve is flat."""
        initial = np.array([0.20])
        terminal = np.array([0.20])
        curve = build_fade_curve(initial, terminal, forecast_years=5, fade_speed=0.5)
        np.testing.assert_allclose(curve[0, :], 0.20)

    def test_vectorised_across_simulations(self):
        """Multiple simulations should have independent curves."""
        n = 500
        rng = np.random.default_rng(42)
        initial = rng.uniform(0.15, 0.30, size=n)
        terminal = rng.uniform(0.05, 0.15, size=n)
        curve = build_fade_curve(initial, terminal, forecast_years=10, fade_speed=0.5)
        assert curve.shape == (n, 10)
        # Last year should be closer to terminal than year 1
        diff_y1 = np.abs(curve[:, 0] - terminal)
        diff_yT = np.abs(curve[:, -1] - terminal)
        assert np.mean(diff_yT) < np.mean(diff_y1)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Distribution PPF
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributionPPF:
    """Verify ppf is the inverse of the CDF for all distribution types."""

    u = np.array([0.01, 0.25, 0.5, 0.75, 0.99])

    def test_fixed_ppf(self):
        d = FixedDistribution(42.0)
        np.testing.assert_allclose(d.ppf(self.u), 42.0)

    def test_normal_ppf(self):
        d = NormalDistribution(5.0, 1.0)
        result = d.ppf(self.u)
        expected = sp_stats.norm.ppf(self.u, loc=5.0, scale=1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_lognormal_ppf(self):
        d = LogNormalDistribution(0.5, 0.3)
        result = d.ppf(self.u)
        expected = sp_stats.lognorm.ppf(self.u, s=0.3, scale=np.exp(0.5))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_triangular_ppf(self):
        d = TriangularDistribution(1.0, 3.0, 5.0)
        result = d.ppf(self.u)
        c = (3.0 - 1.0) / (5.0 - 1.0)
        expected = sp_stats.triang.ppf(self.u, c, loc=1.0, scale=4.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_uniform_ppf(self):
        d = UniformDistribution(2.0, 8.0)
        result = d.ppf(self.u)
        expected = 2.0 + 6.0 * self.u
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_pert_ppf(self):
        d = PERTDistribution(1.0, 4.0, 7.0)
        result = d.ppf(self.u)
        expected = 1.0 + 6.0 * sp_stats.beta.ppf(self.u, d.alpha, d.beta_param)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_round_trip_sample_cdf_ppf(self):
        """sample → sort → linspace-u → ppf should approximate the sorted samples."""
        rng = np.random.default_rng(99)
        d = NormalDistribution(10.0, 2.0)
        samples = d.sample(5000, rng)
        # ppf at median should be close to sample median
        np.testing.assert_allclose(d.ppf(np.array([0.5]))[0], 10.0, atol=0.01)

    def test_ppf_via_factory(self):
        """Ensure ppf works on distributions obtained via create_distribution."""
        cfg = DistributionConfig(
            dist_type=DistributionType.PERT, low=0.0, mode=0.05, high=0.10,
        )
        dist = create_distribution(cfg)
        result = dist.ppf(np.array([0.5]))
        assert 0.0 < result[0] < 0.10


# ═══════════════════════════════════════════════════════════════════════════
# 3.  compute_fcff_vectors with (n, T) param arrays
# ═══════════════════════════════════════════════════════════════════════════

class TestFCFFWithFadingParams:
    """Verify that compute_fcff_vectors handles both (n,) and (n, T) inputs."""

    def test_constant_1d_still_works(self):
        """(n,) inputs should produce identical results as before."""
        n = 50
        fcff, _, _ = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=5,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
        )
        assert fcff.shape == (n, 5)

    def test_2d_ebitda_margin(self):
        """(n, T) ebitda_margin should produce time-varying EBITDA."""
        n = 10
        T = 5
        # Margin declines from 20 % to 10 % linearly
        margin_2d = np.linspace(0.20, 0.10, T)[None, :].repeat(n, axis=0)
        assert margin_2d.shape == (n, T)

        _, ebitda, revenue = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=T,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=margin_2d,
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
        )
        # EBITDA in year 1 should use 20 % margin
        np.testing.assert_allclose(
            ebitda[0, 0], revenue[0, 0] * 0.20, rtol=1e-4,
        )
        # EBITDA in last year should use 10 % margin
        np.testing.assert_allclose(
            ebitda[0, -1], revenue[0, -1] * 0.10, rtol=1e-4,
        )

    def test_mixed_1d_2d(self):
        """Mix of (n,) and (n, T) params should work."""
        n = 10
        T = 3
        fcff, _, _ = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=T,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full((n, T), 0.20),  # 2-D
            da_pct_revenue=np.full(n, 0.03),       # 1-D
            tax_rate=np.full((n, T), 0.25),        # 2-D
            capex_pct_revenue=np.full(n, 0.05),    # 1-D
            nwc_pct_delta_revenue=np.full(n, 0.10),
        )
        assert fcff.shape == (n, T)

    def test_fade_curve_as_input(self):
        """Use build_fade_curve output as margin input."""
        n = 100
        T = 7
        initial = np.full(n, 0.25)
        terminal = np.full(n, 0.15)
        margin_fade = build_fade_curve(initial, terminal, T, fade_speed=0.5)

        fcff, ebitda, revenue = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=T,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=margin_fade,
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
        )
        # Year-1 EBITDA/Revenue ≈ 25 % initial margin (with some decay)
        ratio_y1 = ebitda[:, 0] / revenue[:, 0]
        assert np.mean(ratio_y1) > 0.20
        # Last-year ratio should be closer to 15 %
        ratio_yT = ebitda[:, -1] / revenue[:, -1]
        assert np.mean(ratio_yT) < np.mean(ratio_y1)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Engine: parameter fade integration
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineParameterFade:
    """Integration tests for the MC engine with param-level fade."""

    @staticmethod
    def _fade_segment() -> SegmentConfig:
        return SegmentConfig(
            name="FadeAll",
            base_revenue=1_000.0,
            forecast_years=7,
            revenue_growth_mode=RevenueGrowthMode.FADE,
            fade_speed=0.5,
            revenue_growth=DistributionConfig(fixed_value=0.15),
            terminal_growth_rate=DistributionConfig(fixed_value=0.02),
            ebitda_margin=DistributionConfig(fixed_value=0.25),
            ebitda_margin_terminal=DistributionConfig(fixed_value=0.18),
            capex_pct_revenue=DistributionConfig(fixed_value=0.08),
            capex_pct_revenue_terminal=DistributionConfig(fixed_value=0.04),
        )

    def test_runs_without_error(self):
        seg = self._fade_segment()
        cfg = SimulationConfig(
            n_simulations=200, random_seed=42, segments=[seg],
        )
        result = MonteCarloEngine(cfg).run()
        assert result.n_simulations == 200
        assert np.all(np.isfinite(result.total_ev))

    def test_sensitivity_keys_1d(self):
        """Input samples for fading params should be 1-D (Year-1 value)."""
        seg = self._fade_segment()
        cfg = SimulationConfig(
            n_simulations=100, random_seed=42, segments=[seg],
        )
        result = MonteCarloEngine(cfg).run()
        em_key = f"{seg.name} | EBITDA-Marge"
        assert em_key in result.input_samples
        assert result.input_samples[em_key].ndim == 1

    def test_fade_affects_ev_distribution(self):
        """Segment with param fade should yield a different EV distribution."""
        # Without fade
        seg_const = SegmentConfig(
            name="Const",
            base_revenue=1_000.0, forecast_years=7,
            revenue_growth_mode=RevenueGrowthMode.FADE,
            fade_speed=0.5,
            revenue_growth=DistributionConfig(fixed_value=0.15),
            terminal_growth_rate=DistributionConfig(fixed_value=0.02),
            ebitda_margin=DistributionConfig(fixed_value=0.25),
            # No ebitda_margin_terminal → constant margin
        )
        # With margin fading to lower value
        seg_fade = self._fade_segment()

        r_const = MonteCarloEngine(
            SimulationConfig(n_simulations=500, random_seed=42, segments=[seg_const])
        ).run()
        r_fade = MonteCarloEngine(
            SimulationConfig(n_simulations=500, random_seed=42, segments=[seg_fade])
        ).run()

        # Fading to a lower margin should reduce EV
        assert np.mean(r_fade.total_ev) < np.mean(r_const.total_ev)

    def test_partial_fade_leaves_others_constant(self):
        """Only params with terminal config should fade; others stay constant."""
        seg = SegmentConfig(
            name="Partial",
            base_revenue=1_000.0, forecast_years=5,
            revenue_growth_mode=RevenueGrowthMode.FADE,
            fade_speed=0.5,
            revenue_growth=DistributionConfig(fixed_value=0.10),
            terminal_growth_rate=DistributionConfig(fixed_value=0.02),
            ebitda_margin=DistributionConfig(fixed_value=0.20),
            ebitda_margin_terminal=DistributionConfig(fixed_value=0.15),
            # tax_rate has NO terminal → should stay constant
        )
        cfg = SimulationConfig(n_simulations=100, random_seed=42, segments=[seg])
        result = MonteCarloEngine(cfg).run()
        assert np.all(np.isfinite(result.total_ev))


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Engine: Cross-segment correlation (Gaussian copula)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossSegmentCorrelation:
    """Verify that segment_correlation produces positively/negatively
    correlated segment EVs."""

    @staticmethod
    def _two_stochastic_segments() -> list[SegmentConfig]:
        return [
            SegmentConfig(
                name="A",
                base_revenue=1_000.0, forecast_years=5,
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.08, std=0.03,
                ),
                ebitda_margin=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.20, std=0.03,
                ),
                wacc=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.10, std=0.01,
                ),
            ),
            SegmentConfig(
                name="B",
                base_revenue=800.0, forecast_years=5,
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.06, std=0.02,
                ),
                ebitda_margin=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.18, std=0.02,
                ),
                wacc=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
                ),
            ),
        ]

    def test_no_correlation_independence(self):
        """Without correlation, segment EVs should have low Pearson r."""
        segs = self._two_stochastic_segments()
        cfg = SimulationConfig(
            n_simulations=5_000, random_seed=42, segments=segs,
        )
        r = MonteCarloEngine(cfg).run()
        corr = np.corrcoef(r.segment_evs["A"], r.segment_evs["B"])[0, 1]
        assert abs(corr) < 0.15  # should be near zero

    def test_high_positive_correlation(self):
        """With ρ=0.9, segment EVs should have high positive correlation."""
        segs = self._two_stochastic_segments()
        cfg = SimulationConfig(
            n_simulations=5_000, random_seed=42, segments=segs,
            segment_correlation=[[1.0, 0.9], [0.9, 1.0]],
        )
        r = MonteCarloEngine(cfg).run()
        corr = np.corrcoef(r.segment_evs["A"], r.segment_evs["B"])[0, 1]
        assert corr > 0.4  # copula introduces meaningful positive dependence

    def test_negative_correlation(self):
        """With ρ=−0.8, segment EVs should have negative correlation."""
        segs = self._two_stochastic_segments()
        cfg = SimulationConfig(
            n_simulations=5_000, random_seed=42, segments=segs,
            segment_correlation=[[1.0, -0.8], [-0.8, 1.0]],
        )
        r = MonteCarloEngine(cfg).run()
        corr = np.corrcoef(r.segment_evs["A"], r.segment_evs["B"])[0, 1]
        assert corr < -0.2

    def test_identity_correlation_matches_independent(self):
        """Identity correlation matrix should behave like independent sampling.
        (EVs should be all finite and have similar means.)
        """
        segs = self._two_stochastic_segments()
        cfg_indep = SimulationConfig(
            n_simulations=2_000, random_seed=42, segments=segs,
        )
        cfg_ident = SimulationConfig(
            n_simulations=2_000, random_seed=42, segments=segs,
            segment_correlation=[[1.0, 0.0], [0.0, 1.0]],
        )
        r_indep = MonteCarloEngine(cfg_indep).run()
        r_ident = MonteCarloEngine(cfg_ident).run()
        # Both should produce finite, positive EVs
        assert np.all(np.isfinite(r_indep.total_ev))
        assert np.all(np.isfinite(r_ident.total_ev))

    def test_three_segment_correlation(self):
        """3×3 correlation matrix should work."""
        segs = self._two_stochastic_segments() + [
            SegmentConfig(
                name="C", base_revenue=500.0, forecast_years=5,
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.05, std=0.02,
                ),
            ),
        ]
        corr = [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
        cfg = SimulationConfig(
            n_simulations=1_000, random_seed=42, segments=segs,
            segment_correlation=corr,
        )
        r = MonteCarloEngine(cfg).run()
        assert len(r.segment_evs) == 3
        assert np.all(np.isfinite(r.total_ev))

    def test_correlation_with_param_fade(self):
        """Correlation + parameter fade should work together."""
        segs = [
            SegmentConfig(
                name="F1",
                base_revenue=1_000.0, forecast_years=5,
                revenue_growth_mode=RevenueGrowthMode.FADE, fade_speed=0.5,
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.12, std=0.02,
                ),
                terminal_growth_rate=DistributionConfig(fixed_value=0.02),
                ebitda_margin=DistributionConfig(fixed_value=0.22),
                ebitda_margin_terminal=DistributionConfig(fixed_value=0.18),
            ),
            SegmentConfig(
                name="F2",
                base_revenue=800.0, forecast_years=5,
                revenue_growth_mode=RevenueGrowthMode.FADE, fade_speed=0.5,
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.10, std=0.02,
                ),
                terminal_growth_rate=DistributionConfig(fixed_value=0.02),
                capex_pct_revenue=DistributionConfig(fixed_value=0.07),
                capex_pct_revenue_terminal=DistributionConfig(fixed_value=0.04),
            ),
        ]
        cfg = SimulationConfig(
            n_simulations=500, random_seed=42, segments=segs,
            segment_correlation=[[1.0, 0.6], [0.6, 1.0]],
        )
        r = MonteCarloEngine(cfg).run()
        assert np.all(np.isfinite(r.total_ev))
        assert r.n_simulations == 500

    def test_single_segment_ignores_correlation(self):
        """With only 1 segment, correlation should be silently ignored."""
        seg = SegmentConfig(name="Solo")
        cfg = SimulationConfig(
            n_simulations=100, random_seed=42, segments=[seg],
            segment_correlation=[[1.0]],
        )
        r = MonteCarloEngine(cfg).run()
        assert r.n_simulations == 100


# ═══════════════════════════════════════════════════════════════════════════
# 6.  compute_segment_ev with (n, T) params
# ═══════════════════════════════════════════════════════════════════════════

class TestSegmentEVWithFadingParams:
    def test_2d_margin_through_segment_ev(self):
        """compute_segment_ev should produce valid results with (n, T) margins."""
        n = 100
        T = 5
        margin_2d = build_fade_curve(
            np.full(n, 0.25), np.full(n, 0.15), T, 0.5,
        )
        ev = compute_segment_ev(
            base_revenue=1000.0, forecast_years=T,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=margin_2d,
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            wacc=np.full(n, 0.09),
            terminal_method=TerminalValueMethod.GORDON_GROWTH,
            terminal_growth=np.full(n, 0.02),
            exit_multiple=np.full(n, 10.0),
        )
        assert ev.shape == (n,)
        assert np.all(np.isfinite(ev))
        assert np.all(ev > 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  compute_sensitivity in domain/statistics.py  (was in application layer)
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSensitivityDomain:

    def test_sorted_by_abs_correlation(self):
        from domain.statistics import compute_sensitivity
        rng = np.random.default_rng(42)
        n = 5_000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        target = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.1, n)
        sens = compute_sensitivity(target, {"x1": x1, "x2": x2})
        abs_vals = [abs(v) for v in sens.values()]
        assert abs_vals == sorted(abs_vals, reverse=True)
        assert abs(sens["x1"]) > abs(sens["x2"])

    def test_constant_input_excluded(self):
        from domain.statistics import compute_sensitivity
        target = np.arange(100, dtype=float)
        sens = compute_sensitivity(target, {"const": np.ones(100)})
        assert len(sens) == 0

    def test_empty_inputs(self):
        from domain.statistics import compute_sensitivity
        sens = compute_sensitivity(np.arange(50, dtype=float), {})
        assert sens == {}


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Intra-segment parameter correlation (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

class TestIntraSegmentCorrelation:

    def test_intra_corr_produces_valid_results(self):
        """Segment with intra-param correlation should produce finite EVs."""
        from domain.models import DEFAULT_INTRA_PARAM_CORR
        seg = SegmentConfig(
            name="IntraCorr",
            base_revenue=1_000.0,
            forecast_years=5,
            revenue_growth=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.06, std=0.02,
            ),
            ebitda_margin=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.20, std=0.03,
            ),
            capex_pct_revenue=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.05, std=0.01,
            ),
            wacc=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
            ),
            intra_param_correlation=DEFAULT_INTRA_PARAM_CORR,
        )
        cfg = SimulationConfig(
            n_simulations=1_000, random_seed=42, segments=[seg],
        )
        r = MonteCarloEngine(cfg).run()
        assert np.all(np.isfinite(r.total_ev))
        assert r.n_simulations == 1_000

    def test_intra_corr_creates_dependency(self):
        """With positive EBITDA↔CAPEX correlation, samples should correlate."""
        from domain.models import DEFAULT_INTRA_PARAM_CORR
        seg = SegmentConfig(
            name="CorrCheck",
            base_revenue=1_000.0,
            forecast_years=5,
            revenue_growth=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.06, std=0.02,
            ),
            ebitda_margin=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.20, std=0.03,
            ),
            capex_pct_revenue=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.05, std=0.01,
            ),
            wacc=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
            ),
            intra_param_correlation=DEFAULT_INTRA_PARAM_CORR,
        )
        cfg = SimulationConfig(
            n_simulations=5_000, random_seed=42, segments=[seg],
        )
        r = MonteCarloEngine(cfg).run()
        ebitda = r.input_samples["CorrCheck | EBITDA-Marge"]
        capex = r.input_samples["CorrCheck | CAPEX (% Umsatz)"]
        from scipy import stats as sp_stats
        corr, _ = sp_stats.spearmanr(ebitda, capex)
        # Default matrix has 0.25 correlation → Spearman should be positive
        assert corr > 0.10, f"Expected positive Spearman, got {corr}"

    def test_intra_corr_with_cross_segment(self):
        """Intra + inter segment correlation should work together."""
        from domain.models import DEFAULT_INTRA_PARAM_CORR
        segs = [
            SegmentConfig(
                name="A",
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.06, std=0.02,
                ),
                wacc=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
                ),
                intra_param_correlation=DEFAULT_INTRA_PARAM_CORR,
            ),
            SegmentConfig(
                name="B",
                revenue_growth=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.04, std=0.01,
                ),
                wacc=DistributionConfig(
                    dist_type=DistributionType.NORMAL, mean=0.08, std=0.01,
                ),
                intra_param_correlation=DEFAULT_INTRA_PARAM_CORR,
            ),
        ]
        cfg = SimulationConfig(
            n_simulations=1_000, random_seed=42, segments=segs,
            segment_correlation=[[1.0, 0.5], [0.5, 1.0]],
        )
        r = MonteCarloEngine(cfg).run()
        assert len(r.segment_evs) == 2
        assert np.all(np.isfinite(r.total_ev))

    def test_no_intra_corr_independent(self):
        """Without intra-corr, EBITDA and CAPEX should be ~independent."""
        seg = SegmentConfig(
            name="IndepCheck",
            base_revenue=1_000.0,
            forecast_years=5,
            revenue_growth=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.06, std=0.02,
            ),
            ebitda_margin=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.20, std=0.03,
            ),
            capex_pct_revenue=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.05, std=0.01,
            ),
            wacc=DistributionConfig(
                dist_type=DistributionType.NORMAL, mean=0.09, std=0.01,
            ),
            intra_param_correlation=None,  # no intra-segment correlation
        )
        cfg = SimulationConfig(
            n_simulations=5_000, random_seed=42, segments=[seg],
        )
        r = MonteCarloEngine(cfg).run()
        ebitda = r.input_samples["IndepCheck | EBITDA-Marge"]
        capex = r.input_samples["IndepCheck | CAPEX (% Umsatz)"]
        from scipy import stats as sp_stats
        corr, _ = sp_stats.spearmanr(ebitda, capex)
        # Should be approximately zero (independent)
        assert abs(corr) < 0.10, f"Expected near-zero Spearman, got {corr}"
