"""Tests for application.scenario_service (synthetic configs only)."""
from __future__ import annotations

import pytest

from application.scenario_service import (
    named_steps_for, _p, apply_ops, group_scenarios, reverse_dcf, run_scenarios, run_state, walk,
)

PERT, NORM, FEST = "PERT-Verteilung", "Normalverteilung", "Fest (Deterministisch)"


def _state(**over):
    s = {
        "setup_n_sim": 400, "setup_seed": 7, "setup_n_seg": 1, "setup_ext_bridge": False,
        "bridge_cc_dtype": FEST, "bridge_cc_fixed": 10.0, "bridge_cd_dtype": FEST, "bridge_cd_fixed": 8.0,
        "bridge_nd_dtype": FEST, "bridge_nd_fixed": 100.0, "bridge_sh_dtype": FEST, "bridge_sh_fixed": 10.0,
        "seg_0_name": "A", "seg_0_basrev": 1000.0,
        "s0_em_dtype": PERT, "s0_em_pert_lo": 18.0, "s0_em_pert_mo": 20.0, "s0_em_pert_hi": 23.0,
        "s0_wacc_dtype": NORM, "s0_wacc_n_mu": 8.0, "s0_wacc_n_sig": 0.5,
        "s0_tvg_dtype": FEST, "s0_tvg_fixed": 2.0,
    }
    s.update(over)
    return s


def _cfg(state):
    """Minimal saved config (v2) around a flat state."""
    return {"schema_version": 2, "ui_state": {
        "setup": {k: v for k, v in state.items() if k.startswith("setup_") and k != "setup_ext_bridge"},
        "bridge": {k: v for k, v in state.items() if k.startswith("bridge_") or k == "setup_ext_bridge"},
        "segments": [{k: v for k, v in state.items() if k.startswith(("seg_0", "s0_"))}],
        "correlation": {}}}


class TestApplyOps:
    def test_shift_moves_all_location_values_not_spread(self):
        s, noise = apply_ops(_state(), [{"muster": "s*_em", "op": "+", "wert": 1},
                                        {"muster": "s0_wacc", "op": "+", "wert": 2}])
        assert (s["s0_em_pert_lo"], s["s0_em_pert_mo"], s["s0_em_pert_hi"]) == (19.0, 21.0, 24.0)
        assert (s["s0_wacc_n_mu"], s["s0_wacc_n_sig"]) == (10.0, 0.5)
        assert noise is False

    def test_set_on_prefix_centres_and_keeps_spread(self):
        s, _ = apply_ops(_state(), [{"muster": "s0_em", "op": "=", "wert": 30}])
        assert (s["s0_em_pert_lo"], s["s0_em_pert_mo"], s["s0_em_pert_hi"]) == (28.0, 30.0, 33.0)

    def test_set_exact_key(self):
        s, _ = apply_ops(_state(), [{"muster": "seg_0_basrev", "op": "=", "wert": "1200"}])
        assert s["seg_0_basrev"] == 1200.0

    def test_order_violation_raises(self):
        with pytest.raises(ValueError, match="Min ≤ Modus ≤ Max"):
            apply_ops(_state(), [{"muster": "s0_em_pert_lo", "op": "=", "wert": 25}])

    def test_unknown_pattern_raises_but_valid_new_key_is_created(self):
        with pytest.raises(ValueError, match="trifft kein Feld"):
            apply_ops(_state(), [{"muster": "s0_wac", "op": "+", "wert": 1}])
        with pytest.raises(ValueError, match="trifft kein Feld"):
            apply_ops(_state(), [{"muster": "s0_wac_pert_mo", "op": "=", "wert": 1}])
        s, noise = apply_ops(_state(), [{"muster": "bridge_mi_dtype", "op": "=", "wert": FEST},
                                        {"muster": "bridge_mi_fixed", "op": "=", "wert": 5}])
        assert s["bridge_mi_fixed"] == 5.0 and noise is True

    def test_dtype_change_flags_noise(self):
        _, noise = apply_ops(_state(), [{"muster": "s0_tvg_dtype", "op": "=", "wert": NORM},
                                        {"muster": "s0_tvg_n_mu", "op": "=", "wert": 2}])
        assert noise is True


class TestScenarios:
    def test_group_rows(self):
        rows = [{"szenario": "x", "muster": "s0_em", "op": "+", "wert": 1},
                {"szenario": "y", "muster": "s0_wacc", "op": "+", "wert": 1},
                {"szenario": "x", "muster": "s0_wacc", "op": "+", "wert": -1},
                {"szenario": "", "muster": "s0_em", "op": "+", "wert": 9}]
        g = group_scenarios(rows)
        assert [s["name"] for s in g] == ["x", "y"] and len(g[0]["ops"]) == 2

    def test_delta_against_base_and_error_row(self):
        res = run_scenarios(_state(), [{"name": "WACC +1", "ops": [{"muster": "s0_wacc", "op": "+", "wert": 1}]},
                                       {"name": "kaputt", "ops": [{"muster": "nix", "op": "+", "wert": 1}]}])
        up, bad = res["zeilen"]
        assert up["delta"] < 0 and up["p50"] == pytest.approx(res["basis"]["p50"] + up["delta"])
        assert bad["p50"] is None and "trifft kein Feld" in bad["fehler"]


class TestReverseDcf:
    def test_finds_known_shift(self):
        s = _state()
        st, _ = apply_ops(s, [{"muster": "s0_wacc", "op": "+", "wert": 1.5}])
        target = _p(run_state(st)[1])[1]
        r = reverse_dcf(s, ["s0_wacc"], target, 0.0, 4.0, tol=0.001)
        assert r["status"] == "ok" and r["delta"] == pytest.approx(1.5, abs=0.002)
        assert r["implizit"]["A · wacc"] == pytest.approx(9.5, abs=0.002)

    def test_out_of_range(self):
        r = reverse_dcf(_state(), ["s0_wacc"], 1e9, 0.0, 4.0)
        assert r["status"] == "nicht erreichbar"

    def test_tvg_bound_keeps_gap_to_wacc(self):
        # WACC 8 %, TV-g 2 %: shift may go at most to 8 − 0.5 − 2 = 5.5 pp
        r = reverse_dcf(_state(), ["s0_tvg"], -1e9, -1.0, 20.0)
        assert r["status"] == "nicht erreichbar" and "+5.50" in r["detail"]


class TestWalk:
    def test_endpoints_and_closure(self):
        a = _state()
        b = _state(bridge_nd_fixed=150.0, s0_em_pert_mo=21.0, setup_n_sim=400)
        w = walk(_cfg(a), _cfg(b))
        rows = w["zeilen"]
        assert [r["schritt"] for r in rows] == ["A auf heutigem Modell", "Bridge", "Segment 1"]
        assert rows[0]["p50"] == pytest.approx(_p(run_state(a)[1])[1])
        assert w["schliesst"] and rows[-1]["p50"] == pytest.approx(w["p50_b"])

    def test_both_ends_use_b_iterations_and_seed(self):
        # Like the Feb file (100k) against a 50k file: setup keys are not assumptions.
        a, b = _state(setup_n_sim=300, setup_seed=1), _state(bridge_nd_fixed=150.0)
        w = walk(_cfg(a), _cfg(b))
        assert w["zeilen"][0]["p50"] == pytest.approx(_p(run_state(a, n=400, seed=7)[1])[1])
        assert w["schliesst"]

    def test_method_change_is_its_own_step(self):
        a, b = _state(setup_mid_year=False), _state(setup_mid_year=True)
        w = walk(_cfg(a), _cfg(b))
        assert [r["schritt"] for r in w["zeilen"]] == ["A auf heutigem Modell", "Setup"] and w["schliesst"]

    def test_named_step_first(self):
        a, b = _state(), _state(bridge_nd_fixed=150.0)
        w = walk(_cfg(a), _cfg(b), [{"name": "Schulden", "ops": [{"muster": "bridge_nd", "op": "=", "wert": 150}]}])
        assert [r["schritt"] for r in w["zeilen"]] == ["A auf heutigem Modell", "Schulden"] and w["schliesst"]


def test_named_steps_only_from_their_start_version():
    a = {"metadata": {"stichtag": "2026-02-22"}}
    other = {"saved_at": "2026-11-04T10:00:00"}
    b = {"ueberleitung": [{"name": "Feb", "von": "2026-02-22", "ops": []}, {"name": "immer", "ops": []}]}
    assert [s["name"] for s in named_steps_for(a, b)] == ["Feb", "immer"]
    assert [s["name"] for s in named_steps_for(other, b)] == ["immer"]
