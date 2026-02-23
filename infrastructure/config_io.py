"""
Configuration Serialisation (Save / Load).

Provides helpers to serialise and deserialise the Streamlit
session-state used by ``app.py``.  All functions accept and
return plain dicts / values – they are **not** coupled to
``streamlit.session_state`` directly, making them testable
without a running Streamlit process.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np


# ── Key registries (shared with the Streamlit widgets) ────────────────

SETUP_KEYS: list[str] = [
    "setup_n_sim", "setup_seed", "setup_n_seg",
    "setup_corp_costs", "setup_corp_disc", "setup_net_debt", "setup_shares",
]

DIST_PARAMS: list[str] = [
    "rg", "em", "da", "tx", "cx", "nwc", "wacc", "tvg", "evm",
]

DIST_SUFFIXES: list[str] = [
    "_dtype", "_fixed",
    "_n_mu", "_n_sig",
    "_ln_mu", "_ln_sig",
    "_tri_lo", "_tri_mo", "_tri_hi",
    "_uni_lo", "_uni_hi",
    "_pert_lo", "_pert_mo", "_pert_hi",
]

_SEG_PATTERN = re.compile(r"^(seg_\d+_|s\d+_)")


# ── Serialisation ─────────────────────────────────────────────────────

def _coerce_json(v):
    """Convert numpy scalars to native Python types for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def collect_config(state: dict) -> dict:
    """Build a JSON-serialisable config dict from a state mapping.

    Parameters
    ----------
    state : dict
        Typically ``dict(st.session_state)`` – the full Streamlit
        session state snapshot.

    Returns
    -------
    dict
        Versioned configuration ready for ``json.dumps``.
    """
    cfg: dict = {"version": 1, "saved_at": datetime.now().isoformat()}

    # Setup parameters
    setup: dict = {}
    for k in SETUP_KEYS:
        if k in state:
            v = state[k]
            setup[k] = int(v) if isinstance(v, (int, np.integer)) else float(v)
    cfg["setup"] = setup

    # Per-segment parameters
    n_seg = int(state.get("setup_n_seg", 1))
    segments: list[dict] = []
    for i in range(n_seg):
        seg: dict = {}
        for suffix in ["_name", "_basrev", "_fyrs", "_tv_method"]:
            key = f"seg_{i}{suffix}"
            if key in state:
                seg[key] = _coerce_json(state[key])
        for param in DIST_PARAMS:
            prefix = f"s{i}_{param}"
            for sfx in DIST_SUFFIXES:
                key = f"{prefix}{sfx}"
                if key in state:
                    seg[key] = _coerce_json(state[key])
        segments.append(seg)
    cfg["segments"] = segments
    return cfg


def apply_config(cfg: dict, state: dict) -> dict:
    """Merge a loaded config into a state dict (non-destructive return).

    Parameters
    ----------
    cfg : dict
        Previously saved configuration (from ``collect_config``).
    state : dict
        Current session state.  A shallow copy is mutated and returned.

    Returns
    -------
    dict
        Updated state mapping with stale segment keys removed and
        new values applied.
    """
    updated = dict(state)

    # Clear stale segment keys so that old segments don't persist
    for k in list(updated.keys()):
        if _SEG_PATTERN.match(k):
            del updated[k]

    # Apply setup keys
    for k, v in cfg.get("setup", {}).items():
        updated[k] = v

    # Apply segment keys
    for seg_data in cfg.get("segments", []):
        for k, v in seg_data.items():
            updated[k] = v

    return updated
