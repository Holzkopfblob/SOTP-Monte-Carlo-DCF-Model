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
        assert cfg["version"] == 1
        assert "saved_at" in cfg

    def test_setup_section(self):
        cfg = collect_config(_build_state())
        setup = cfg["setup"]
        assert setup["setup_n_sim"] == 10_000
        assert setup["setup_seed"] == 42
        assert setup["setup_mid_year"] is True
        assert isinstance(setup["setup_n_sim"], int)

    def test_bridge_section(self):
        cfg = collect_config(_build_state())
        bridge = cfg["bridge"]
        assert bridge["bridge_cc_dtype"] == "Fest (Deterministisch)"
        assert bridge["bridge_cc_fixed"] == 50.0

    def test_segment_count_matches(self):
        for n in (1, 3):
            cfg = collect_config(_build_state(n_segments=n))
            assert len(cfg["segments"]) == n

    def test_segment_contains_expected_keys(self):
        cfg = collect_config(_build_state(1))
        seg = cfg["segments"][0]
        assert seg["seg_0_name"] == "Segment 0"
        assert seg["seg_0_basrev"] == 1_000.0
        assert seg["s0_rg_dtype"] == "Normal"

    def test_growth_mode_saved(self):
        cfg = collect_config(_build_state(1, with_fade=True))
        seg = cfg["segments"][0]
        assert seg["seg_0_growth_mode"] == "Fade-Modell (g konvergiert zum Terminal-Wachstum)"
        assert seg["seg_0_fade_speed"] == pytest.approx(0.7)
        assert seg["seg_0_param_fade"] is True

    def test_constant_growth_mode_saved(self):
        cfg = collect_config(_build_state(1))
        seg = cfg["segments"][0]
        assert seg["seg_0_growth_mode"] == "Konstant (g \u00fcber alle Jahre gleich)"

    def test_sampling_method_saved(self):
        cfg = collect_config(_build_state(1))
        assert cfg["setup"]["setup_sampling"] == "Pseudo-Random (Standard)"

    def test_cross_segment_correlation_saved(self):
        cfg = collect_config(_build_state(2, with_corr=True))
        assert cfg["setup"]["setup_corr_enable"] is True
        corr = cfg["correlation"]
        assert corr["corr_0_0"] == 1.0
        assert corr["corr_0_1"] == 0.5
        assert corr["corr_1_0"] == 0.5
        assert corr["corr_1_1"] == 1.0

    def test_intra_segment_correlation_saved(self):
        cfg = collect_config(_build_state(1, with_intra_corr=True))
        seg = cfg["segments"][0]
        assert seg["seg_0_intra_corr"] is True
        assert seg["seg_0_ic_0_0"] == 1.0
        assert seg["seg_0_ic_0_1"] == 0.3

    def test_terminal_dist_params_saved(self):
        state = _build_state(1, with_fade=True)
        # Add a terminal distribution param
        state["s0_em_term_dtype"] = "Fest (Deterministisch)"
        state["s0_em_term_fixed"] = np.float64(0.18)
        cfg = collect_config(state)
        seg = cfg["segments"][0]
        assert seg["s0_em_term_dtype"] == "Fest (Deterministisch)"
        assert seg["s0_em_term_fixed"] == pytest.approx(0.18)

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
