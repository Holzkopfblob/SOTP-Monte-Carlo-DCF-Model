"""
Tests for infrastructure.config_io – config serialisation round-trip.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from infrastructure.config_io import (
    DIST_PARAMS,
    DIST_SUFFIXES,
    SETUP_KEYS,
    apply_config,
    collect_config,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _build_state(n_segments: int = 1) -> dict:
    """Build a fake session-state dict resembling Streamlit widget values."""
    state: dict = {
        "setup_n_sim": np.int64(10_000),
        "setup_seed": np.int64(42),
        "setup_n_seg": np.int64(n_segments),
        "setup_corp_costs": np.float64(50.0),
        "setup_corp_disc": np.float64(0.09),
        "setup_net_debt": np.float64(200.0),
        "setup_shares": np.float64(100.0),
    }
    for i in range(n_segments):
        state[f"seg_{i}_name"] = f"Segment {i}"
        state[f"seg_{i}_basrev"] = np.float64(1_000.0 * (i + 1))
        state[f"seg_{i}_fyrs"] = np.int64(5)
        state[f"seg_{i}_tv_method"] = "Gordon Growth"
        for param in DIST_PARAMS:
            prefix = f"s{i}_{param}"
            state[f"{prefix}_dtype"] = "Normal"
            state[f"{prefix}_fixed"] = np.float64(0.05)
            state[f"{prefix}_n_mu"] = np.float64(0.05)
            state[f"{prefix}_n_sig"] = np.float64(0.01)
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
        assert isinstance(setup["setup_n_sim"], int)

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
        # All segment keys should match
        for i in range(n_seg):
            for suffix in ["_name", "_basrev", "_fyrs", "_tv_method"]:
                key = f"seg_{i}{suffix}"
                assert restored.get(key) == state.get(key), f"{key} mismatch"
