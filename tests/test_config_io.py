"""
Tests for infrastructure.config_io – config serialisation round-trip.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from infrastructure.config_io import (
    BRIDGE_PREFIXES,
    DIST_PARAMS,
    DIST_SUFFIXES,
    SETUP_KEYS,
    apply_config,
    collect_config,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _build_state(n_segments: int = 1, *, with_fade: bool = False,
                 with_corr: bool = False, with_intra_corr: bool = False) -> dict:
    """Build a fake session-state dict resembling Streamlit widget values."""
    state: dict = {
        "setup_n_sim": np.int64(10_000),
        "setup_seed": np.int64(42),
        "setup_n_seg": np.int64(n_segments),
        "setup_mid_year": True,
        "setup_ext_bridge": False,
        "setup_sampling": "Pseudo-Random (Standard)",
        "setup_corr_enable": with_corr,
    }
    # Bridge distribution widget keys
    for prefix in BRIDGE_PREFIXES:
        state[f"{prefix}_dtype"] = "Fest (Deterministisch)"
        state[f"{prefix}_fixed"] = np.float64(50.0)
    for i in range(n_segments):
        state[f"seg_{i}_name"] = f"Segment {i}"
        state[f"seg_{i}_basrev"] = np.float64(1_000.0 * (i + 1))
        state[f"seg_{i}_fyrs"] = np.int64(5)
        state[f"seg_{i}_tv_method"] = "Gordon Growth"
        if with_fade:
            state[f"seg_{i}_growth_mode"] = "Fade-Modell (g konvergiert zum Terminal-Wachstum)"
            state[f"seg_{i}_fade_speed"] = 0.7
            state[f"seg_{i}_param_fade"] = True
        else:
            state[f"seg_{i}_growth_mode"] = "Konstant (g \u00fcber alle Jahre gleich)"
        if with_intra_corr:
            state[f"seg_{i}_intra_corr"] = True
            for row in range(7):
                for col_idx in range(7):
                    state[f"seg_{i}_ic_{row}_{col_idx}"] = (
                        1.0 if row == col_idx else 0.3
                    )
        for param in DIST_PARAMS:
            prefix = f"s{i}_{param}"
            state[f"{prefix}_dtype"] = "Normal"
            state[f"{prefix}_fixed"] = np.float64(0.05)
            state[f"{prefix}_n_mu"] = np.float64(0.05)
            state[f"{prefix}_n_sig"] = np.float64(0.01)
    if with_corr and n_segments >= 2:
        for row in range(n_segments):
            for col_idx in range(n_segments):
                state[f"corr_{row}_{col_idx}"] = (
                    1.0 if row == col_idx else 0.5
                )
    return state


# ═══════════════════════════════════════════════════════════════════════════
# collect_config
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectConfig:
    def test_returns_versioned_dict(self):
        cfg = collect_config(_build_state())
        assert cfg["schema_version"] == 2
        assert "saved_at" in cfg
        assert "ui_state" in cfg

    def test_setup_section(self):
        cfg = collect_config(_build_state())
        setup = cfg["ui_state"]["setup"]
        assert setup["setup_n_sim"] == 10_000
        assert setup["setup_seed"] == 42
        assert setup["setup_mid_year"] is True
        assert isinstance(setup["setup_n_sim"], int)

    def test_bridge_section(self):
        cfg = collect_config(_build_state())
        bridge = cfg["ui_state"]["bridge"]
        assert bridge["bridge_cc_dtype"] == "Fest (Deterministisch)"
        assert bridge["bridge_cc_fixed"] == 50.0

    def test_segment_count_matches(self):
        for n in (1, 3):
            cfg = collect_config(_build_state(n_segments=n))
            assert len(cfg["ui_state"]["segments"]) == n

    def test_segment_contains_expected_keys(self):
        cfg = collect_config(_build_state(1))
        seg = cfg["ui_state"]["segments"][0]
        assert seg["seg_0_name"] == "Segment 0"
        assert seg["seg_0_basrev"] == 1_000.0
        assert seg["s0_rg_dtype"] == "Normal"

    def test_growth_mode_saved(self):
        cfg = collect_config(_build_state(1, with_fade=True))
        seg = cfg["ui_state"]["segments"][0]
        assert seg["seg_0_growth_mode"] == "Fade-Modell (g konvergiert zum Terminal-Wachstum)"
        assert seg["seg_0_fade_speed"] == pytest.approx(0.7)
        assert seg["seg_0_param_fade"] is True

    def test_constant_growth_mode_saved(self):
        cfg = collect_config(_build_state(1))
        seg = cfg["ui_state"]["segments"][0]
        assert seg["seg_0_growth_mode"] == "Konstant (g \u00fcber alle Jahre gleich)"

    def test_sampling_method_saved(self):
        cfg = collect_config(_build_state(1))
        assert cfg["ui_state"]["setup"]["setup_sampling"] == "Pseudo-Random (Standard)"

    def test_cross_segment_correlation_saved(self):
        cfg = collect_config(_build_state(2, with_corr=True))
        assert cfg["ui_state"]["setup"]["setup_corr_enable"] is True
        corr = cfg["ui_state"]["correlation"]
        assert corr["corr_0_0"] == 1.0
        assert corr["corr_0_1"] == 0.5
        assert corr["corr_1_0"] == 0.5
        assert corr["corr_1_1"] == 1.0

    def test_intra_segment_correlation_saved(self):
        cfg = collect_config(_build_state(1, with_intra_corr=True))
        seg = cfg["ui_state"]["segments"][0]
        assert seg["seg_0_intra_corr"] is True
        assert seg["seg_0_ic_0_0"] == 1.0
        assert seg["seg_0_ic_0_1"] == 0.3

    def test_terminal_dist_params_saved(self):
        state = _build_state(1, with_fade=True)
        # Add a terminal distribution param
        state["s0_em_term_dtype"] = "Fest (Deterministisch)"
        state["s0_em_term_fixed"] = np.float64(0.18)
        cfg = collect_config(state)
        seg = cfg["ui_state"]["segments"][0]
        assert seg["s0_em_term_dtype"] == "Fest (Deterministisch)"
        assert seg["s0_em_term_fixed"] == pytest.approx(0.18)

    def test_simulation_metadata_section(self):
        cfg = collect_config(_build_state(2, with_corr=True))
        sim = cfg["simulation"]
        assert sim["n_simulations"] == 10_000
        assert sim["random_seed"] == 42
        assert sim["n_segments"] == 2
        assert sim["segment_correlation_enabled"] is True

    def test_numpy_coerced_to_native(self):
        """All numpy scalars must become plain int / float for JSON."""
        cfg = collect_config(_build_state())
        blob = json.dumps(cfg)  # would fail if numpy types are left
        assert isinstance(blob, str)

    def test_extra_keys_ignored(self):
        """Keys outside the known registries are silently skipped."""
        state = _build_state()
        state["_internal_flag"] = True  # not a setup or segment key
        cfg = collect_config(state)
        assert "_internal_flag" not in json.dumps(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# apply_config
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyConfig:
    def test_setup_keys_restored(self):
        cfg = collect_config(_build_state())
        updated = apply_config(cfg, {})
        assert updated["setup_n_sim"] == 10_000
        assert updated["setup_seed"] == 42

    def test_segment_keys_restored(self):
        cfg = collect_config(_build_state(2))
        updated = apply_config(cfg, {})
        assert updated["seg_0_name"] == "Segment 0"
        assert updated["seg_1_name"] == "Segment 1"

    def test_stale_segment_keys_cleared(self):
        """When loading a 1-segment config on top of 2-segment state, old seg_1 keys vanish."""
        old_state = _build_state(n_segments=2)
        new_cfg = collect_config(_build_state(n_segments=1))
        updated = apply_config(new_cfg, old_state)
        # seg_1 keys should be gone
        assert "seg_1_name" not in updated
        assert "s1_rg_dtype" not in updated
        # seg_0 keys should be present
        assert "seg_0_name" in updated
        # bridge keys should be restored
        assert "bridge_cc_dtype" in updated

    def test_stale_corr_keys_cleared(self):
        """When loading config without correlation on top of state with correlation."""
        old_state = _build_state(n_segments=2, with_corr=True)
        new_cfg = collect_config(_build_state(n_segments=1))
        updated = apply_config(new_cfg, old_state)
        assert "corr_0_1" not in updated
        assert "corr_1_0" not in updated

    def test_non_segment_keys_preserved(self):
        """Keys unrelated to segments survive the apply."""
        old_state = {"_my_custom_flag": True}
        cfg = collect_config(_build_state(1))
        updated = apply_config(cfg, old_state)
        assert updated["_my_custom_flag"] is True

    def test_original_state_not_mutated(self):
        original = _build_state()
        cfg = collect_config(_build_state(1))
        before_keys = set(original.keys())
        apply_config(cfg, original)
        assert set(original.keys()) == before_keys

    def test_legacy_v1_payload_still_supported(self):
        legacy_cfg = {
            "version": 1,
            "setup": {"setup_n_sim": 1_234, "setup_seed": 7},
            "bridge": {"bridge_cc_dtype": "Fest (Deterministisch)", "bridge_cc_fixed": 50.0},
            "segments": [{"seg_0_name": "Legacy Segment"}],
            "correlation": {"corr_0_0": 1.0},
        }
        updated = apply_config(legacy_cfg, {})
        assert updated["setup_n_sim"] == 1_234
        assert updated["setup_seed"] == 7
        assert updated["seg_0_name"] == "Legacy Segment"
        assert updated["corr_0_0"] == pytest.approx(1.0)

    def test_oldest_v1_bridge_in_setup_is_migrated(self):
        # Early v1 files (Feb 2026) kept the bridge as plain numbers in "setup".
        # Without migration they silently fell back to the defaults (net debt 500, 100 shares).
        from infrastructure.config_io import build_simulation_config
        old_cfg = {
            "version": 1,
            "setup": {"setup_n_sim": 100, "setup_seed": 42, "setup_n_seg": 1,
                      "setup_corp_costs": 800.0, "setup_corp_disc": 6.0,
                      "setup_net_debt": 137_300.0, "setup_shares": 4_938.0},
            "segments": [{"seg_0_name": "Old", "seg_0_basrev": 1_000.0}],
        }
        b = build_simulation_config(old_cfg).corporate_bridge
        assert b.annual_corporate_costs == pytest.approx(800.0)
        assert b.corporate_cost_discount_rate == pytest.approx(0.06)
        assert b.net_debt == pytest.approx(137_300.0)
        assert b.shares_outstanding == pytest.approx(4_938.0)

    def test_explicit_bridge_wins_over_oldest_setup_keys(self):
        cfg = {
            "version": 1,
            "setup": {"setup_net_debt": 1.0},
            "bridge": {"bridge_nd_dtype": "Fest (Deterministisch)", "bridge_nd_fixed": 2.0},
        }
        assert apply_config(cfg, {})["bridge_nd_fixed"] == pytest.approx(2.0)

    def test_scenario_blocks_round_trip_and_results_only_when_current(self):
        from types import SimpleNamespace
        from infrastructure.config_io import RESULTS_FINGERPRINT_KEY, SCENARIO_RESULTS_KEY, ui_fingerprint
        defs = [{"name": "WACC +1", "ops": [{"muster": "s*_wacc", "op": "+", "wert": 1}]}]
        cfg = collect_config(_build_state(1))
        cfg["szenarien"] = defs
        cfg["ueberleitung"] = [{"name": "x", "von": "2026-02-22", "ops": []}]
        state = apply_config(cfg, {})
        arr = np.array([1.0, 2.0, 3.0])
        state["results"] = SimpleNamespace(n_simulations=3, total_ev=arr, equity_values=arr, price_per_share=arr)
        state[RESULTS_FINGERPRINT_KEY] = ui_fingerprint(state)
        state[SCENARIO_RESULTS_KEY] = {"fingerprint": ui_fingerprint(state), "definitionen": defs,
                                       "szenarien": {"zeilen": []}, "reverse_dcf": {"status": "ok"}}
        saved = collect_config(state)
        assert saved["szenarien"] == defs and saved["ueberleitung"][0]["von"] == "2026-02-22"
        assert saved["ergebnis"]["szenarien"] == {"zeilen": []}
        assert saved["ergebnis"]["reverse_dcf"] == {"status": "ok"}
        # definitions edited after the run → stale results are not saved
        state["cfg_passthrough"]["szenarien"] = [{"name": "anders", "ops": []}]
        assert "szenarien" not in collect_config(state)["ergebnis"]

    def test_unknown_top_level_blocks_are_preserved(self):
        cfg = collect_config(_build_state(1))
        cfg["metadata"] = {"report_currency": "EUR", "company": "Airbus SE"}
        cfg["notes"] = {"source": "research"}
        updated = apply_config(cfg, {})
        assert "cfg_passthrough" in updated
        assert updated["cfg_passthrough"]["metadata"]["report_currency"] == "EUR"
        assert updated["cfg_passthrough"]["notes"]["source"] == "research"


# ═══════════════════════════════════════════════════════════════════════════
# Round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    @pytest.mark.parametrize("n_seg", [1, 2, 3])
    def test_collect_then_apply_is_identity(self, n_seg):
        """Serialize → JSON → deserialize → apply → compare relevant keys."""
        state = _build_state(n_segments=n_seg)
        cfg = collect_config(state)
        blob = json.dumps(cfg)
        loaded = json.loads(blob)
        restored = apply_config(loaded, {})
        # All setup keys should match
        for k in SETUP_KEYS:
            if k in state:
                assert restored[k] == pytest.approx(state[k]), f"{k} mismatch"
        # All bridge keys should match
        for prefix in BRIDGE_PREFIXES:
            for sfx in DIST_SUFFIXES:
                key = f"{prefix}{sfx}"
                if key in state:
                    assert restored[key] == pytest.approx(state[key]), f"{key} mismatch"
        # All segment keys should match
        for i in range(n_seg):
            for suffix in ["_name", "_basrev", "_fyrs", "_tv_method",
                           "_growth_mode", "_fade_speed", "_param_fade",
                           "_intra_corr"]:
                key = f"seg_{i}{suffix}"
                if key in state:
                    assert restored.get(key) == state.get(key), f"{key} mismatch"

    def test_roundtrip_fade_mode(self):
        """Fade growth mode and speed survive a round-trip."""
        state = _build_state(1, with_fade=True)
        cfg = collect_config(state)
        blob = json.dumps(cfg)
        loaded = json.loads(blob)
        restored = apply_config(loaded, {})
        assert restored["seg_0_growth_mode"] == state["seg_0_growth_mode"]
        assert restored["seg_0_fade_speed"] == pytest.approx(state["seg_0_fade_speed"])
        assert restored["seg_0_param_fade"] is True

    def test_roundtrip_cross_segment_correlation(self):
        """Cross-segment correlation survives a round-trip."""
        state = _build_state(2, with_corr=True)
        cfg = collect_config(state)
        blob = json.dumps(cfg)
        loaded = json.loads(blob)
        restored = apply_config(loaded, {})
        assert restored["setup_corr_enable"] is True
        assert restored["corr_0_1"] == pytest.approx(0.5)
        assert restored["corr_1_0"] == pytest.approx(0.5)

    def test_roundtrip_intra_segment_correlation(self):
        """Intra-segment correlation survives a round-trip."""
        state = _build_state(1, with_intra_corr=True)
        cfg = collect_config(state)
        blob = json.dumps(cfg)
        loaded = json.loads(blob)
        restored = apply_config(loaded, {})
        assert restored["seg_0_intra_corr"] is True
        assert restored["seg_0_ic_0_0"] == pytest.approx(1.0)
        assert restored["seg_0_ic_0_1"] == pytest.approx(0.3)
        assert restored["seg_0_ic_3_5"] == pytest.approx(0.3)

    def test_roundtrip_preserves_unknown_top_level_fields(self):
        state = _build_state(1)
        loaded_cfg = collect_config(state)
        loaded_cfg["metadata"] = {"report_currency": "EUR", "sector_tags": ["Industrie"]}
        loaded_cfg["research"] = {"version": "2026-03-29"}

        applied = apply_config(loaded_cfg, {})
        re_export = collect_config(applied)

        assert re_export["metadata"]["report_currency"] == "EUR"
        assert re_export["metadata"]["sector_tags"] == ["Industrie"]
        assert re_export["research"]["version"] == "2026-03-29"


# ═══════════════════════════════════════════════════════════════════════════
# Research import, result snapshot, version diff
# ═══════════════════════════════════════════════════════════════════════════

from pathlib import Path
from types import SimpleNamespace

from infrastructure.config_io import (
    RESULTS_FINGERPRINT_KEY,
    diff_configs,
    parse_config_text,
    ui_fingerprint,
)

_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "sotp_research_prompt.md"


class TestResearchImport:
    def test_prompt_template_is_importable(self):
        """The JSON template in the research prompt must load into the app."""
        cfg = parse_config_text(_PROMPT.read_text(encoding="utf-8"))
        state = apply_config(cfg, {})
        assert state["s0_rg_pert_mo"] == pytest.approx(5.0)
        assert state["cfg_passthrough"]["quellen"]["s0_rg"].endswith("[Q2]")
        assert "Q1" in state["cfg_passthrough"]["quellenverzeichnis"]

    def test_extracts_fenced_block_from_llm_chatter(self):
        text = 'Checks {ok}\n```json\n{"ui_state": {"setup": {"setup_n_seg": 2}}}\n```\nEnde {x}'
        assert parse_config_text(text)["ui_state"]["setup"]["setup_n_seg"] == 2

    @pytest.mark.parametrize("text", ["kein json", "{nicht: gültig}", '{"foo": 1}'])
    def test_rejects_unusable_text(self, text):
        with pytest.raises(ValueError):
            parse_config_text(text)


class TestResultSnapshot:
    def _results(self):
        arr = np.linspace(0.0, 100.0, 101)
        return SimpleNamespace(n_simulations=101, total_ev=arr, equity_values=arr, price_per_share=arr)

    def test_snapshot_saved_only_for_matching_inputs(self):
        state = _build_state(1)
        state["results"] = self._results()
        state[RESULTS_FINGERPRINT_KEY] = ui_fingerprint(state)

        snap = collect_config(state)["ergebnis"]
        assert snap["preis_je_aktie"]["p50"] == pytest.approx(50.0)
        assert snap["preis_je_aktie"]["p5"] == pytest.approx(5.0)

        state["seg_0_basrev"] = 9_999.0  # input edited after the run
        assert "ergebnis" not in collect_config(state)

    def test_loaded_snapshot_is_not_passed_through(self):
        cfg = collect_config(_build_state(1))
        cfg["ergebnis"] = {"stale": True}
        assert "ergebnis" not in collect_config(apply_config(cfg, {}))


class TestDiffConfigs:
    def test_reports_changes_with_labels_and_sources(self):
        state = _build_state(2, with_corr=True)
        state["s0_rg_dtype"] = state["s0_wacc_dtype"] = "Fest (Deterministisch)"
        old = collect_config(state)
        new = json.loads(json.dumps(old))
        seg = new["ui_state"]["segments"][0]
        seg["s0_rg_fixed"] = 7.0
        seg["s0_wacc_n_mu"] = 1.0  # leftover of an inactive dist type → ignored
        new["ui_state"]["correlation"].update(corr_0_1=0.8, corr_1_0=0.8)
        new["quellen"] = {"s0_rg": "Konsens [Q2]"}

        rows = diff_configs(old, new)
        assert [(r["parameter"], r["alt"], r["neu"]) for r in rows] == [
            ("Umsatzwachstum (%) · Wert", 0.05, 7.0),
            ("ρ Segment 1 ↔ 2", 0.5, 0.8),
        ]
        assert rows[0]["quelle"] == "Konsens [Q2]"
        assert diff_configs(old, old) == []

        # Value changed again, justification copied over unchanged → flagged.
        newer = json.loads(json.dumps(new))
        newer["ui_state"]["segments"][0]["s0_rg_fixed"] = 8.0
        assert diff_configs(new, newer)[0]["quelle"].startswith("⚠️")
