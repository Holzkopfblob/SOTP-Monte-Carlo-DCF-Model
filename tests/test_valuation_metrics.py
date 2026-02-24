"""
Tests for Phase 2 – Valuation Quality Metrics.

Covers domain/valuation_metrics.py functions and their integration
with the Monte-Carlo engine (SimulationResults fields).
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.valuation_metrics import (
    _nopat_margin,
    _reinvest_margin,
    implied_roic,
    reinvestment_rate,
    tv_ev_ratio,
    valuation_quality_score,
    _score_convergence,
    _score_dispersion,
    _score_sensitivity,
    _score_tv_ev,
)


# ═══════════════════════════════════════════════════════════════════════════
# tv_ev_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestTvEvRatio:

    def test_basic_ratio(self):
        """TV / EV for deterministic values."""
        pv_tv = np.array([70.0, 80.0])
        ev = np.array([100.0, 100.0])
        ratio = tv_ev_ratio(pv_tv, ev)
        np.testing.assert_allclose(ratio, [0.70, 0.80])

    def test_clamps_to_zero_one(self):
        """Negative TV or TV > EV should be clamped."""
        pv_tv = np.array([-10.0, 150.0])
        ev = np.array([100.0, 100.0])
        ratio = tv_ev_ratio(pv_tv, ev)
        assert ratio[0] == 0.0
        assert ratio[1] == 1.0

    def test_zero_ev_safe(self):
        """EV ≈ 0 should not cause division by zero."""
        pv_tv = np.array([50.0])
        ev = np.array([0.0])
        ratio = tv_ev_ratio(pv_tv, ev)
        assert np.isfinite(ratio[0])

    def test_vector_shapes(self):
        """Output shape must match input."""
        n = 500
        pv_tv = np.random.default_rng(1).uniform(20, 80, n)
        ev = np.random.default_rng(2).uniform(80, 120, n)
        ratio = tv_ev_ratio(pv_tv, ev)
        assert ratio.shape == (n,)
        assert np.all((ratio >= 0) & (ratio <= 1))


# ═══════════════════════════════════════════════════════════════════════════
# Quality score sub-components
# ═══════════════════════════════════════════════════════════════════════════

class TestQualitySubScores:

    def test_tv_ev_full_score(self):
        assert _score_tv_ev(0.30) == 25.0  # well below 40 %

    def test_tv_ev_zero_score(self):
        assert _score_tv_ev(0.95) == 0.0  # above 90 %

    def test_convergence_full(self):
        assert _score_convergence(0.1) == pytest.approx(25.0, abs=0.5)

    def test_convergence_zero(self):
        assert _score_convergence(6.0) == 0.0

    def test_sensitivity_deterministic(self):
        """No correlations → full score (deterministic model)."""
        assert _score_sensitivity({}) == 25.0

    def test_sensitivity_single_driver(self):
        """One dominant driver → HHI = 1 → zero score."""
        assert _score_sensitivity({"x": 0.95}) == 0.0

    def test_sensitivity_diversified(self):
        """Many equal drivers → low HHI → high score."""
        corr = {f"x{i}": 0.3 for i in range(10)}
        score = _score_sensitivity(corr)
        assert score > 15.0  # should be well above half

    def test_dispersion_low_cv(self):
        assert _score_dispersion(0.05) == pytest.approx(25.0, abs=0.5)

    def test_dispersion_high_cv(self):
        assert _score_dispersion(1.5) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Composite quality score
# ═══════════════════════════════════════════════════════════════════════════

class TestValuationQualityScore:

    def test_returns_all_keys(self):
        q = valuation_quality_score(0.5, 1.0, {"a": 0.5, "b": 0.3}, 1000.0, 100.0)
        assert set(q.keys()) == {"total", "tv_ev", "convergence", "sensitivity", "dispersion"}

    def test_total_is_sum(self):
        q = valuation_quality_score(0.6, 2.0, {"a": 0.5}, 500.0, 50.0)
        expected = q["tv_ev"] + q["convergence"] + q["sensitivity"] + q["dispersion"]
        assert q["total"] == pytest.approx(expected, abs=0.01)

    def test_perfect_model(self):
        """All dimensions ideal → score near 100."""
        q = valuation_quality_score(
            mean_tv_ev=0.30,
            ci_width_pct=0.1,
            correlations={f"x{i}": 0.2 for i in range(15)},
            equity_mean=1000.0,
            equity_std=50.0,
        )
        assert q["total"] > 85.0

    def test_poor_model(self):
        """All dimensions bad → score near 0."""
        q = valuation_quality_score(
            mean_tv_ev=0.95,
            ci_width_pct=10.0,
            correlations={"only_driver": 0.99},
            equity_mean=100.0,
            equity_std=200.0,
        )
        assert q["total"] < 15.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_segment_ev decompose=True
# ═══════════════════════════════════════════════════════════════════════════

class TestSegmentEVDecompose:

    def test_decompose_returns_detail(self):
        from domain.valuation import compute_segment_ev
        n = 100
        g = np.full(n, 0.05)
        detail = compute_segment_ev(
            base_revenue=1000.0,
            forecast_years=5,
            revenue_growth=g,
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            wacc=np.full(n, 0.09),
            terminal_method="Gordon Growth Model (Ewige Rente)",
            terminal_growth=np.full(n, 0.02),
            exit_multiple=np.ones(n),
            decompose=True,
        )
        assert hasattr(detail, "ev")
        assert hasattr(detail, "pv_fcff")
        assert hasattr(detail, "pv_tv")
        assert detail.ev.shape == (n,)
        np.testing.assert_allclose(detail.ev, detail.pv_fcff + detail.pv_tv)

    def test_decompose_false_returns_array(self):
        from domain.valuation import compute_segment_ev
        n = 50
        result = compute_segment_ev(
            base_revenue=1000.0,
            forecast_years=5,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            wacc=np.full(n, 0.09),
            terminal_method="Gordon Growth Model (Ewige Rente)",
            terminal_growth=np.full(n, 0.02),
            exit_multiple=np.ones(n),
            decompose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: SimulationResults has new fields
# ═══════════════════════════════════════════════════════════════════════════

class TestEnginePhase2Integration:

    def test_results_have_tv_ev(self, minimal_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(minimal_sim_config)
        assert len(results.segment_tv_ev_ratios) == 1
        for arr in results.segment_tv_ev_ratios.values():
            assert arr.shape == (minimal_sim_config.n_simulations,)
            assert np.all((arr >= 0) & (arr <= 1))

    def test_results_have_quality_score(self, minimal_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(minimal_sim_config)
        q = results.quality_score
        assert "total" in q
        assert 0 <= q["total"] <= 100

    def test_multi_segment_tv_ev(self, full_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(full_sim_config)
        assert len(results.segment_tv_ev_ratios) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Chart factories (smoke tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase2Charts:

    def test_tv_ev_chart(self):
        from presentation.charts import tv_ev_decomposition_chart
        fig = tv_ev_decomposition_chart(
            ["Seg A", "Seg B"], [0.30, 0.25], [0.70, 0.75],
        )
        assert fig is not None

    def test_quality_gauge(self):
        from presentation.charts import quality_score_gauge
        fig = quality_score_gauge({"total": 72.5})
        assert fig is not None

    def test_quality_breakdown(self):
        from presentation.charts import quality_score_breakdown_chart
        fig = quality_score_breakdown_chart({
            "tv_ev": 20, "convergence": 22, "sensitivity": 15, "dispersion": 18,
        })
        assert fig is not None


# ═══════════════════════════════════════════════════════════════════════════
# Shared margin helpers  (DRY extraction)
# ═══════════════════════════════════════════════════════════════════════════

class TestMarginHelpers:

    def test_nopat_margin(self):
        n = 100
        nopat = _nopat_margin(
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
        )
        np.testing.assert_allclose(nopat, 0.1275)

    def test_reinvest_margin(self):
        n = 100
        rm = _reinvest_margin(
            capex_pct_revenue=np.full(n, 0.05),
            da_pct_revenue=np.full(n, 0.03),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            revenue_growth=np.full(n, 0.05),
        )
        # 0.05 - 0.03 + 0.10 * 0.05/1.05 ≈ 0.0248
        np.testing.assert_allclose(rm, 0.02 + 0.10 * 0.05 / 1.05, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Implied ROIC
# ═══════════════════════════════════════════════════════════════════════════

class TestImpliedROIC:

    def test_basic_positive_roic(self):
        """Typical profitable company should have positive ROIC.

        With the corrected formula ROIC = g × NOPAT_margin / reinvest_margin:
        NOPAT margin = (0.20 - 0.03) × 0.75 = 0.1275
        Reinvest margin = 0.05 - 0.03 + 0.10 × 0.05/1.05 ≈ 0.02476
        ROIC = 0.05 × 0.1275 / 0.02476 ≈ 0.257  (25.7 %)
        """
        n = 1000
        roic = implied_roic(
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            revenue_growth=np.full(n, 0.05),
        )
        assert roic.shape == (n,)
        assert np.all(roic > 0)
        # Should be ≈ 25.7 %, well within realistic range
        np.testing.assert_allclose(roic[0], 0.257, atol=0.01)

    def test_clamped_range(self):
        """Output is clamped to [-1, 2]."""
        n = 100
        roic = implied_roic(
            ebitda_margin=np.full(n, 0.50),
            da_pct_revenue=np.full(n, 0.01),
            tax_rate=np.full(n, 0.10),
            capex_pct_revenue=np.full(n, 0.02),
            nwc_pct_delta_revenue=np.full(n, 0.01),
            revenue_growth=np.full(n, 0.03),
        )
        assert np.all(roic >= -1.0)
        assert np.all(roic <= 2.0)

    def test_zero_growth_gives_zero_roic(self):
        """When g = 0, ROIC = g × ... = 0 (no growth → no capital turnover)."""
        n = 50
        roic = implied_roic(
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            revenue_growth=np.full(n, 0.0),
        )
        np.testing.assert_allclose(roic, 0.0, atol=1e-6)

    def test_vector_shapes(self):
        n = 500
        rng = np.random.default_rng(42)
        roic = implied_roic(
            ebitda_margin=rng.uniform(0.10, 0.30, n),
            da_pct_revenue=rng.uniform(0.02, 0.05, n),
            tax_rate=rng.uniform(0.20, 0.30, n),
            capex_pct_revenue=rng.uniform(0.03, 0.08, n),
            nwc_pct_delta_revenue=rng.uniform(0.05, 0.15, n),
            revenue_growth=rng.uniform(0.02, 0.10, n),
        )
        assert roic.shape == (n,)
        assert np.all(np.isfinite(roic))


class TestReinvestmentRate:

    def test_basic_positive(self):
        """Typical case: positive reinvestment."""
        n = 100
        rr = reinvestment_rate(
            capex_pct_revenue=np.full(n, 0.05),
            da_pct_revenue=np.full(n, 0.03),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
            tax_rate=np.full(n, 0.25),
        )
        assert rr.shape == (n,)
        assert np.all(rr > 0)  # net capex + NWC > 0

    def test_clamped_range(self):
        n = 100
        rr = reinvestment_rate(
            capex_pct_revenue=np.full(n, 0.01),
            da_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.01),
            revenue_growth=np.full(n, 0.03),
            ebitda_margin=np.full(n, 0.20),
            tax_rate=np.full(n, 0.25),
        )
        assert np.all(rr >= -1.0)
        assert np.all(rr <= 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Engine populates ROIC & reinvestment fields
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineROICIntegration:

    def test_results_have_roic(self, minimal_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(minimal_sim_config)
        assert len(results.segment_implied_roic) == 1
        for arr in results.segment_implied_roic.values():
            assert arr.shape == (minimal_sim_config.n_simulations,)
            assert np.all(np.isfinite(arr))

    def test_results_have_reinvest_rates(self, minimal_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(minimal_sim_config)
        assert len(results.segment_reinvest_rates) == 1
        for arr in results.segment_reinvest_rates.values():
            assert arr.shape == (minimal_sim_config.n_simulations,)

    def test_multi_segment_roic(self, full_sim_config):
        from application.simulation_service import SimulationService
        results = SimulationService.run_simulation(full_sim_config)
        assert len(results.segment_implied_roic) == 2
        assert len(results.segment_reinvest_rates) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ROIC chart smoke tests
# ═══════════════════════════════════════════════════════════════════════════

class TestROICCharts:

    def test_roic_histogram(self):
        from presentation.charts import roic_histogram
        data = {"Seg A": np.random.default_rng(1).uniform(0.1, 0.3, 1000)}
        fig = roic_histogram(data, wacc_mean=0.09)
        assert fig is not None

    def test_roic_histogram_no_wacc(self):
        from presentation.charts import roic_histogram
        data = {"Seg A": np.random.default_rng(1).normal(0.15, 0.05, 500)}
        fig = roic_histogram(data)
        assert fig is not None

    def test_reinvestment_rate_chart(self):
        from presentation.charts import reinvestment_rate_chart
        data = {"Seg A": np.random.default_rng(1).uniform(0.1, 0.5, 500)}
        fig = reinvestment_rate_chart(data)
        assert fig is not None

    def test_roic_vs_wacc_scatter(self):
        from presentation.charts import roic_vs_wacc_scatter
        rng = np.random.default_rng(42)
        roic = {"S1": rng.uniform(0.05, 0.30, 500)}
        wacc = {"S1": rng.uniform(0.07, 0.12, 500)}
        fig = roic_vs_wacc_scatter(roic, wacc)
        assert fig is not None
