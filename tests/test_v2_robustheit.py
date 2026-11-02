"""
Tests for v2 robustness changes.

A1 – state → SimulationConfig in config_io (one conversion for app + headless).
A2 – Gordon terminal value at steady-state (fade-target) parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from domain.fade import build_fade_curve
from domain.models import (
    DistributionType,
    RevenueGrowthMode,
    SimulationConfig,
    TerminalValueMethod,
)
from domain.valuation import compute_segment_ev, steady_state_fcff
from infrastructure.config_io import (
    build_simulation_config,
    dist_config_from_state,
    segment_config_from_state,
    segment_correlation_from_state,
    simulation_config_from_state,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine

REPO = Path(__file__).resolve().parents[1]
FADE = RevenueGrowthMode.FADE.value
CONST = RevenueGrowthMode.CONSTANT.value
EXIT = TerminalValueMethod.EXIT_MULTIPLE.value


# ═══════════════════════════════════════════════════════════════════════════
# A1 – conversion
# ═══════════════════════════════════════════════════════════════════════════

class TestDistConfigFromState:
    def test_percent_fixed_divided_by_100(self):
        c = dist_config_from_state({"p_dtype": "Fest (Deterministisch)", "p_fixed": 8.2}, "p",
                                   is_percentage=True)
        assert c.fixed_value == pytest.approx(0.082)

    def test_absolute_not_divided(self):
        c = dist_config_from_state({"p_dtype": "Fest (Deterministisch)", "p_fixed": 790.0}, "p")
        assert c.fixed_value == 790.0

    def test_pert_percent(self):
        s = {"p_dtype": "PERT-Verteilung", "p_pert_lo": 4.0, "p_pert_mo": 6.5, "p_pert_hi": 10.0}
        c = dist_config_from_state(s, "p", is_percentage=True)
        assert (c.low, c.mode, c.high) == pytest.approx((0.04, 0.065, 0.10))

    def test_lognormal_display_moments_are_kept(self, rng):
        s = {"p_dtype": "Lognormalverteilung", "p_ln_mu": 20.0, "p_ln_sig": 3.0}
        c = dist_config_from_state(s, "p", is_percentage=True)
        draws = rng.lognormal(c.ln_mu, c.ln_sigma, 200_000)
        assert draws.mean() == pytest.approx(0.20, rel=0.01)
        assert draws.std() == pytest.approx(0.03, rel=0.02)

    def test_missing_keys_use_widget_default(self):
        c = dist_config_from_state({}, "p", default=5.0, is_percentage=True)
        assert c.dist_type == DistributionType.FIXED
        assert c.fixed_value == pytest.approx(0.05)


class TestSegmentRules:
    def test_param_fade_ignored_without_fade_mode(self):
        s = {"seg_0_growth_mode": CONST, "seg_0_param_fade": True, "seg_0_fade_speed": 1.5,
             "s0_em_term_dtype": "Fest (Deterministisch)", "s0_em_term_fixed": 30.0}
        seg = segment_config_from_state(s, 0)
        assert seg.ebitda_margin_terminal is None
        assert seg.fade_speed == 0.5

    def test_param_fade_terminals_with_fade_mode(self):
        s = {"seg_0_growth_mode": FADE, "seg_0_param_fade": True, "seg_0_fade_speed": 0.3,
             "s0_cx_term_dtype": "Fest (Deterministisch)", "s0_cx_term_fixed": 14.0}
        seg = segment_config_from_state(s, 0)
        assert seg.capex_pct_revenue_terminal.fixed_value == pytest.approx(0.14)
        assert seg.fade_speed == 0.3

    def test_exit_multiple_without_fade_uses_fixed_growth_and_raw_multiple(self):
        s = {"seg_0_tv_method": EXIT, "seg_0_growth_mode": CONST,
             "s0_evm_dtype": "Fest (Deterministisch)", "s0_evm_fixed": 9.0}
        seg = segment_config_from_state(s, 0)
        assert seg.exit_multiple.fixed_value == 9.0          # multiple, not %
        assert seg.terminal_growth_rate.fixed_value == 0.02

    def test_intra_corr_mirrors_upper_triangle(self):
        s = {"seg_0_intra_corr": True, "seg_0_ic_0_5": 0.0}
        m = segment_config_from_state(s, 0).intra_param_correlation
        assert m[0][5] == m[5][0] == 0.0

    def test_intra_corr_not_psd_is_dropped(self):
        s = {"seg_0_intra_corr": True, "seg_0_ic_0_1": 0.99, "seg_0_ic_0_2": 0.99,
             "seg_0_ic_1_2": -0.99}
        assert segment_config_from_state(s, 0).intra_param_correlation is None

    def test_segment_correlation_off_or_single_segment(self):
        assert segment_correlation_from_state({"setup_corr_enable": False}, 3) is None
        assert segment_correlation_from_state({"setup_corr_enable": True}, 1) is None


class TestBridgeFromState:
    def test_extended_items_zero_when_disabled(self):
        s = {"setup_n_seg": 1, "setup_ext_bridge": False,
             "bridge_pn_dtype": "Fest (Deterministisch)", "bridge_pn_fixed": 391.6}
        b = simulation_config_from_state(s).corporate_bridge
        assert b.pension_liabilities == 0.0 and b.stochastic_pension_liabilities is None

    def test_extended_items_used_when_enabled(self):
        s = {"setup_n_seg": 1, "setup_ext_bridge": True,
             "bridge_pn_dtype": "Fest (Deterministisch)", "bridge_pn_fixed": 391.6,
             "bridge_cd_dtype": "Fest (Deterministisch)", "bridge_cd_fixed": 9.2}
        b = simulation_config_from_state(s).corporate_bridge
        assert b.pension_liabilities == 391.6
        assert b.corporate_cost_discount_rate == pytest.approx(0.092)

    def test_stochastic_item_keeps_distribution(self):
        s = {"setup_n_seg": 1, "bridge_sh_dtype": "PERT-Verteilung",
             "bridge_sh_pert_lo": 100.0, "bridge_sh_pert_mo": 100.7, "bridge_sh_pert_hi": 101.5}
        b = simulation_config_from_state(s).corporate_bridge
        assert b.shares_outstanding == 100.7
        assert b.stochastic_shares.dist_type == DistributionType.PERT


class TestBuildSimulationConfig:
    def test_airbus_example_builds_and_runs(self):
        cfg = json.loads((REPO / "airbus_sotp_config.json").read_text(encoding="utf-8"))
        sim = build_simulation_config(cfg)
        assert isinstance(sim, SimulationConfig)
        assert [s.name for s in sim.segments] == [
            "Airbus Commercial Aircraft", "Airbus Defence and Space", "Airbus Helicopters"]
        assert sim.segments[0].wacc.mean == pytest.approx(0.082)
        assert sim.segment_correlation is not None
        sim.n_simulations = 2_000
        res = MonteCarloEngine(sim).run()
        assert np.all(np.isfinite(res.price_per_share))


# ═══════════════════════════════════════════════════════════════════════════
# A2 – steady-state terminal value
# ═══════════════════════════════════════════════════════════════════════════

def _ev(n=4, T=10, lam=0.3, terminal_params=None, growth_mode=RevenueGrowthMode.FADE,
        g0=0.12, g_tv=0.03, cx0=0.30, cx_t=0.10, method=TerminalValueMethod.GORDON_GROWTH):
    capex = build_fade_curve(np.full(n, cx0), np.full(n, cx_t), T, lam)
    return compute_segment_ev(
        base_revenue=1000.0, forecast_years=T,
        revenue_growth=np.full(n, g0),
        ebitda_margin=np.full(n, 0.45),
        da_pct_revenue=np.full(n, 0.05),
        tax_rate=np.full(n, 0.18),
        capex_pct_revenue=capex,
        nwc_pct_delta_revenue=np.full(n, 0.05),
        wacc=np.full(n, 0.09),
        terminal_method=method,
        terminal_growth=np.full(n, g_tv),
        exit_multiple=np.full(n, 12.0),
        growth_mode=growth_mode, fade_speed=lam,
        decompose=True, terminal_params=terminal_params,
    )


class TestSteadyStateTerminalValue:
    def test_equals_legacy_when_fade_has_converged(self):
        """λ → ∞: year-T values equal the targets, so both TVs coincide."""
        term = {"capex_pct_revenue": np.full(4, 0.10)}
        new = _ev(lam=50.0, terminal_params=term)
        old = _ev(lam=50.0, terminal_params=None)
        np.testing.assert_allclose(new.pv_tv, old.pv_tv, rtol=1e-9)
        np.testing.assert_allclose(new.pv_fcff, old.pv_fcff, rtol=1e-12)

    def test_slow_fade_uses_targets_not_year_t(self):
        term = {"capex_pct_revenue": np.full(4, 0.10)}
        new = _ev(lam=0.3, terminal_params=term)
        old = _ev(lam=0.3, terminal_params=None)
        # Year-T capex is still above 10 % → legacy TV too low
        assert np.all(new.pv_tv > old.pv_tv)
        np.testing.assert_allclose(new.pv_fcff, old.pv_fcff, rtol=1e-12)

    def test_matches_hand_calculation(self):
        """TV = FCFF_{T+1}(targets) / (WACC − g), FCFF chain as in the model."""
        rev_t, g = 2000.0, 0.03
        fcff = steady_state_fcff(np.array([rev_t]), np.array([g]), {
            "ebitda_margin": np.array([0.45]), "da_pct_revenue": np.array([0.05]),
            "tax_rate": np.array([0.18]), "capex_pct_revenue": np.array([0.10]),
            "nwc_pct_delta_revenue": np.array([0.05]),
        })
        rev = rev_t * 1.03
        ebit = rev * 0.40
        expected = ebit * 0.82 + rev * 0.05 - rev * 0.10 - (rev - rev_t) * 0.05
        assert fcff[0] == pytest.approx(expected)

    def test_exit_multiple_ignores_terminal_params(self):
        term = {"capex_pct_revenue": np.full(4, 0.10)}
        a = _ev(method=TerminalValueMethod.EXIT_MULTIPLE, terminal_params=term)
        b = _ev(method=TerminalValueMethod.EXIT_MULTIPLE, terminal_params=None)
        np.testing.assert_allclose(a.ev, b.ev, rtol=1e-12)

    def test_engine_uses_steady_state_tv(self):
        """Deterministic segment: engine EV equals the direct A2 formula."""
        f = "Fest (Deterministisch)"
        s = {"setup_n_seg": 1, "setup_n_sim": 1_000, "setup_mid_year": True,
             "seg_0_basrev": 1000.0, "seg_0_fyrs": 10, "seg_0_growth_mode": FADE,
             "seg_0_fade_speed": 0.3, "seg_0_param_fade": True,
             "s0_rg_dtype": f, "s0_rg_fixed": 12.0, "s0_em_dtype": f, "s0_em_fixed": 45.0,
             "s0_da_dtype": f, "s0_da_fixed": 5.0, "s0_tx_dtype": f, "s0_tx_fixed": 18.0,
             "s0_cx_dtype": f, "s0_cx_fixed": 30.0, "s0_nwc_dtype": f, "s0_nwc_fixed": 5.0,
             "s0_wacc_dtype": f, "s0_wacc_fixed": 9.0, "s0_tvg_dtype": f, "s0_tvg_fixed": 3.0,
             "s0_em_term_dtype": f, "s0_em_term_fixed": 45.0,
             "s0_da_term_dtype": f, "s0_da_term_fixed": 5.0,
             "s0_tx_term_dtype": f, "s0_tx_term_fixed": 18.0,
             "s0_cx_term_dtype": f, "s0_cx_term_fixed": 10.0,
             "s0_nwc_term_dtype": f, "s0_nwc_term_fixed": 5.0}
        res = MonteCarloEngine(simulation_config_from_state(s)).run()
        engine_ev = next(iter(res.segment_evs.values()))
        direct = _ev(n=1, terminal_params={"capex_pct_revenue": np.full(1, 0.10)})
        legacy = _ev(n=1, terminal_params=None)
        assert engine_ev[0] == pytest.approx(direct.ev[0], rel=1e-9)
        assert engine_ev[0] != pytest.approx(legacy.ev[0], rel=1e-6)

    def test_engine_terminals_do_not_leak_into_inputs(self):
        """The internal terminal-draw entry must not show up as an input sample."""
        cfg = json.loads((REPO / "airbus_sotp_config.json").read_text(encoding="utf-8"))
        sim = build_simulation_config(cfg)
        sim.n_simulations = 2_000
        res = MonteCarloEngine(sim).run()
        assert set(res.segment_evs) == {s.name for s in sim.segments}
        assert not any(k.startswith("__") for k in res.input_samples)
