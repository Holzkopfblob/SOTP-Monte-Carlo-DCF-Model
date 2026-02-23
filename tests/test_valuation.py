"""
Tests for domain.valuation – vectorised FCFF, terminal value, segment EV.

Validates financial logic, edge cases, and mathematical correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

from domain.models import RevenueGrowthMode, TerminalValueMethod
from domain.valuation import (
    compute_corporate_costs_pv,
    compute_fcff_vectors,
    compute_segment_ev,
    compute_terminal_value,
)


class TestComputeFCFFVectors:
    """FCFF schedule computation."""

    def test_output_shapes(self):
        n = 100
        fcff, ebitda, revenue = compute_fcff_vectors(
            base_revenue=1000.0,
            forecast_years=5,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
        )
        assert fcff.shape == (n, 5)
        assert ebitda.shape == (n, 5)
        assert revenue.shape == (n, 5)

    def test_constant_growth_revenue(self):
        """Revenue should compound at g each year."""
        n = 1
        g = 0.10
        base = 1000.0
        _, _, revenue = compute_fcff_vectors(
            base_revenue=base, forecast_years=3,
            revenue_growth=np.array([g]),
            ebitda_margin=np.array([0.20]),
            da_pct_revenue=np.array([0.03]),
            tax_rate=np.array([0.25]),
            capex_pct_revenue=np.array([0.05]),
            nwc_pct_delta_revenue=np.array([0.10]),
        )
        for yr in range(3):
            expected = base * (1 + g) ** (yr + 1)
            np.testing.assert_allclose(revenue[0, yr], expected, rtol=1e-10)

    def test_fcff_formula(self):
        """FCFF = NOPAT + D&A - CAPEX - delta_NWC."""
        n = 1
        fcff, ebitda, revenue = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=1,
            revenue_growth=np.array([0.10]),
            ebitda_margin=np.array([0.20]),
            da_pct_revenue=np.array([0.03]),
            tax_rate=np.array([0.25]),
            capex_pct_revenue=np.array([0.05]),
            nwc_pct_delta_revenue=np.array([0.10]),
        )
        rev = 1000.0 * 1.10
        eb = rev * 0.20
        da = rev * 0.03
        ebit = eb - da
        nopat = ebit * (1 - 0.25)
        capex = rev * 0.05
        delta_nwc = (rev - 1000.0) * 0.10
        expected_fcff = nopat + da - capex - delta_nwc
        np.testing.assert_allclose(fcff[0, 0], expected_fcff, rtol=1e-10)

    def test_fade_model_growth_decay(self):
        """Fade model: g should start near g_init and converge to g_term."""
        n = 1
        g_init = 0.20
        g_term = 0.02
        _, _, revenue = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=10,
            revenue_growth=np.array([g_init]),
            ebitda_margin=np.array([0.20]),
            da_pct_revenue=np.array([0.03]),
            tax_rate=np.array([0.25]),
            capex_pct_revenue=np.array([0.05]),
            nwc_pct_delta_revenue=np.array([0.10]),
            growth_mode=RevenueGrowthMode.FADE,
            terminal_growth=np.array([g_term]),
            fade_speed=0.5,
        )
        # Year-1 growth should be close to g_init (with some fade)
        actual_g1 = revenue[0, 0] / 1000.0 - 1.0
        assert actual_g1 > 0.10  # still well above terminal
        # Year-10 growth should be close to g_term
        actual_g10 = revenue[0, 9] / revenue[0, 8] - 1.0
        assert actual_g10 < 0.05  # significantly faded toward 2%

    def test_negative_ebit_no_tax(self):
        """Taxes should be zero when EBIT is negative."""
        n = 1
        fcff, _, _ = compute_fcff_vectors(
            base_revenue=1000.0, forecast_years=1,
            revenue_growth=np.array([0.05]),
            ebitda_margin=np.array([0.02]),  # very low margin
            da_pct_revenue=np.array([0.05]),  # D&A > EBITDA margin → negative EBIT
            tax_rate=np.array([0.30]),
            capex_pct_revenue=np.array([0.01]),
            nwc_pct_delta_revenue=np.array([0.01]),
        )
        rev = 1000.0 * 1.05
        ebitda = rev * 0.02
        da = rev * 0.05
        ebit = ebitda - da
        assert ebit < 0  # confirm EBIT is negative
        # NOPAT = EBIT (no tax on negative)
        nopat = ebit  # taxes = max(ebit,0)*0.3 = 0
        expected = nopat + da - rev * 0.01 - (rev - 1000.0) * 0.01
        np.testing.assert_allclose(fcff[0, 0], expected, rtol=1e-10)


class TestComputeTerminalValue:
    def test_gordon_growth(self):
        fcff_last = np.array([100.0])
        wacc = np.array([0.10])
        g = np.array([0.02])
        tv = compute_terminal_value(
            TerminalValueMethod.GORDON_GROWTH,
            fcff_last, np.array([0.0]), wacc, g, np.array([0.0]),
        )
        expected = 100.0 * 1.02 / (0.10 - 0.02)
        np.testing.assert_allclose(tv[0], expected, rtol=1e-10)

    def test_exit_multiple(self):
        ebitda_last = np.array([200.0])
        mult = np.array([12.0])
        tv = compute_terminal_value(
            TerminalValueMethod.EXIT_MULTIPLE,
            np.array([0.0]), ebitda_last, np.array([0.0]),
            np.array([0.0]), mult,
        )
        np.testing.assert_allclose(tv[0], 2400.0)

    def test_gordon_wacc_equals_g_clamped(self):
        """When WACC ≈ g, denominator must be clamped to avoid infinity."""
        tv = compute_terminal_value(
            TerminalValueMethod.GORDON_GROWTH,
            np.array([100.0]), np.array([0.0]),
            np.array([0.05]), np.array([0.05]),
            np.array([0.0]),
        )
        assert np.isfinite(tv[0])


class TestComputeSegmentEV:
    def test_positive_ev(self):
        n = 500
        ev = compute_segment_ev(
            base_revenue=1000.0, forecast_years=5,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
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
        assert np.all(ev > 0)

    def test_mid_year_convention_higher(self):
        """Mid-year convention should yield higher EV (earlier discounting)."""
        n = 100
        kwargs = dict(
            base_revenue=1000.0, forecast_years=5,
            revenue_growth=np.full(n, 0.05),
            ebitda_margin=np.full(n, 0.20),
            da_pct_revenue=np.full(n, 0.03),
            tax_rate=np.full(n, 0.25),
            capex_pct_revenue=np.full(n, 0.05),
            nwc_pct_delta_revenue=np.full(n, 0.10),
            wacc=np.full(n, 0.09),
            terminal_method=TerminalValueMethod.GORDON_GROWTH,
            terminal_growth=np.full(n, 0.02),
            exit_multiple=np.full(n, 10.0),
        )
        ev_mid = compute_segment_ev(**kwargs, mid_year_convention=True)
        ev_end = compute_segment_ev(**kwargs, mid_year_convention=False)
        assert np.mean(ev_mid) > np.mean(ev_end)


class TestCorporateCostsPV:
    def test_perpetuity_formula(self):
        pv = compute_corporate_costs_pv(50.0, np.array([0.10]))
        np.testing.assert_allclose(pv[0], 500.0)

    def test_vector_costs(self):
        costs = np.array([40.0, 60.0])
        rate = np.array([0.08, 0.10])
        pv = compute_corporate_costs_pv(costs, rate)
        np.testing.assert_allclose(pv, [500.0, 600.0])

    def test_zero_rate_clamped(self):
        pv = compute_corporate_costs_pv(50.0, np.array([0.0]))
        assert np.isfinite(pv[0])
