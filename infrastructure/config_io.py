"""
Configuration Serialisation (Save / Load).

Provides helpers to serialise and deserialise the Streamlit
session-state used by ``app.py``.  All functions accept and
return plain dicts / values – they are **not** coupled to
``streamlit.session_state`` directly, making them testable
without a running Streamlit process.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
from domain.models import INTRA_PARAM_LABELS, CorporateBridgeConfig, DistributionType


# ── Key registries (shared with the Streamlit widgets) ────────────────

SETUP_KEYS: list[str] = [
    "setup_n_sim", "setup_seed", "setup_n_seg",
    "setup_mid_year", "setup_ext_bridge",
    "setup_sampling", "setup_corr_enable",
]

# Derived from CorporateBridgeConfig field names – single source of truth.
_BRIDGE_FIELD_TO_PREFIX = {
    "annual_corporate_costs": "bridge_cc",
    "corporate_cost_discount_rate": "bridge_cd",
    "net_debt": "bridge_nd",
    "shares_outstanding": "bridge_sh",
    "minority_interests": "bridge_mi",
    "pension_liabilities": "bridge_pn",
    "non_operating_assets": "bridge_no",
    "associate_investments": "bridge_as",
}
BRIDGE_PREFIXES: list[str] = list(_BRIDGE_FIELD_TO_PREFIX.values())

DIST_PARAMS: list[str] = [
    "rg", "em", "da", "tx", "cx", "nwc", "wacc", "tvg", "evm",
    # Phase 3: terminal distribution params for parameter fade
    "em_term", "da_term", "tx_term", "cx_term", "nwc_term",
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
_BRIDGE_PATTERN = re.compile(r"^bridge_")
_CORR_PATTERN = re.compile(r"^corr_\d+_\d+$")
_PASSTHROUGH_STATE_KEY = "cfg_passthrough"
_KNOWN_TOP_LEVEL_KEYS = {
    "schema_version",
    "saved_at",
    "simulation",
    "ui_state",
    # Recomputed on every save – a loaded snapshot must never be passed
    # through, or it would be attached to assumptions it does not belong to.
    "ergebnis",
    # legacy v1 keys
    "version",
    "setup",
    "bridge",
    "segments",
    "correlation",
}


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
    cfg: dict = {
        "schema_version": 2,
        "saved_at": datetime.now().isoformat(),
    }

    # Setup parameters
    setup: dict = {}
    for k in SETUP_KEYS:
        if k in state:
            v = state[k]
            if isinstance(v, bool):
                setup[k] = v
            elif isinstance(v, (int, np.integer)):
                setup[k] = int(v)
            elif isinstance(v, str):
                setup[k] = v
            else:
                setup[k] = float(v)
    ui_state: dict = {"setup": setup}

    cfg["simulation"] = {
        "n_simulations": int(setup.get("setup_n_sim", 10_000)),
        "random_seed": int(setup.get("setup_seed", 42)),
        "n_segments": int(setup.get("setup_n_seg", 1)),
        "mid_year_convention": bool(setup.get("setup_mid_year", True)),
        "sampling_method": setup.get("setup_sampling"),
        "segment_correlation_enabled": bool(setup.get("setup_corr_enable", False)),
    }

    # Bridge parameters (distribution inputs)
    bridge: dict = {}
    for prefix in BRIDGE_PREFIXES:
        for sfx in DIST_SUFFIXES:
            key = f"{prefix}{sfx}"
            if key in state:
                bridge[key] = _coerce_json(state[key])
    # Also capture the ext-bridge checkbox
    if "setup_ext_bridge" in state:
        bridge["setup_ext_bridge"] = bool(state["setup_ext_bridge"])
    ui_state["bridge"] = bridge

    # Per-segment parameters
    n_seg = int(state.get("setup_n_seg", 1))
    segments: list[dict] = []
    for i in range(n_seg):
        seg: dict = {}
        for suffix in ["_name", "_basrev", "_fyrs", "_tv_method",
                       "_growth_mode", "_fade_speed", "_param_fade",
                       "_intra_corr"]:
            key = f"seg_{i}{suffix}"
            if key in state:
                seg[key] = _coerce_json(state[key])
        for param in DIST_PARAMS:
            prefix = f"s{i}_{param}"
            for sfx in DIST_SUFFIXES:
                key = f"{prefix}{sfx}"
                if key in state:
                    seg[key] = _coerce_json(state[key])
        # Intra-segment correlation matrix (7×7)
        for row in range(7):
            for col_idx in range(7):
                key = f"seg_{i}_ic_{row}_{col_idx}"
                if key in state:
                    seg[key] = _coerce_json(state[key])
        segments.append(seg)
    ui_state["segments"] = segments

    # Cross-segment correlation matrix
    correlation: dict = {}
    for row in range(n_seg):
        for col_idx in range(n_seg):
            key = f"corr_{row}_{col_idx}"
            if key in state:
                correlation[key] = _coerce_json(state[key])
    ui_state["correlation"] = correlation

    cfg["ui_state"] = ui_state

    # Result snapshot – only if the results were produced from exactly
    # these inputs (fingerprint set by the simulation step).
    results = state.get("results")
    if results is not None and state.get(RESULTS_FINGERPRINT_KEY) == _fingerprint(ui_state):
        cfg["ergebnis"] = summarize_results(results)
        # Scenario / reverse-DCF results only if computed from these inputs and definitions.
        scen = state.get(SCENARIO_RESULTS_KEY) or {}
        defs = (state.get(_PASSTHROUGH_STATE_KEY) or {}).get("szenarien")
        if scen.get("fingerprint") == _fingerprint(ui_state) and scen.get("definitionen") == defs:
            for k in ("szenarien", "reverse_dcf"):
                if scen.get(k) is not None:
                    cfg["ergebnis"][k] = scen[k]

    # Preserve unknown top-level config fields loaded from external files
    # (e.g. metadata, notes, taxonomy). These are not interpreted by the UI
    # but should survive an import -> export round-trip.
    passthrough = state.get(_PASSTHROUGH_STATE_KEY)
    if isinstance(passthrough, dict):
        for key, value in passthrough.items():
            if key not in _KNOWN_TOP_LEVEL_KEYS:
                cfg[key] = value

    return cfg


# Early v1 files kept the bridge as plain numbers in "setup".
_OLDEST_V1_BRIDGE = {"setup_corp_costs": "bridge_cc", "setup_corp_disc": "bridge_cd",
                     "setup_net_debt": "bridge_nd", "setup_shares": "bridge_sh"}


def _extract_ui_state(cfg: dict) -> dict:
    """Return normalized ui_state from v2 schema or legacy v1 payload."""
    if isinstance(cfg.get("ui_state"), dict):
        return cfg["ui_state"]
    setup = cfg.get("setup", {})
    bridge = dict(cfg.get("bridge", {}))
    for old, prefix in _OLDEST_V1_BRIDGE.items():
        if old in setup and f"{prefix}_dtype" not in bridge:
            bridge[f"{prefix}_dtype"] = DistributionType.FIXED.value
            bridge[f"{prefix}_fixed"] = setup[old]
    return {
        "setup": setup,
        "bridge": bridge,
        "segments": cfg.get("segments", []),
        "correlation": cfg.get("correlation", {}),
    }


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

    # Clear stale bridge keys
    for k in list(updated.keys()):
        if _BRIDGE_PATTERN.match(k):
            del updated[k]

    # Clear stale cross-segment correlation keys
    for k in list(updated.keys()):
        if _CORR_PATTERN.match(k):
            del updated[k]

    payload = _extract_ui_state(cfg)

    # Apply setup keys
    for k, v in payload.get("setup", {}).items():
        updated[k] = v

    # Apply bridge keys
    for k, v in payload.get("bridge", {}).items():
        updated[k] = v

    # Apply segment keys
    for seg_data in payload.get("segments", []):
        for k, v in seg_data.items():
            updated[k] = v

    # Apply cross-segment correlation keys
    for k, v in payload.get("correlation", {}).items():
        updated[k] = v

    # Keep unknown top-level payload blocks so they survive a round-trip.
    passthrough: dict = {}
    for k, v in cfg.items():
        if k not in _KNOWN_TOP_LEVEL_KEYS:
            passthrough[k] = v
    if passthrough:
        updated[_PASSTHROUGH_STATE_KEY] = passthrough
    elif _PASSTHROUGH_STATE_KEY in updated:
        del updated[_PASSTHROUGH_STATE_KEY]

    return updated


# ── Result snapshot ───────────────────────────────────────────────────

RESULTS_FINGERPRINT_KEY = "results_fingerprint"
SCENARIO_RESULTS_KEY = "szenario_ergebnis"  # {"fingerprint", "definitionen", "szenarien", "reverse_dcf"}


def _fingerprint(ui_state: dict) -> str:
    return json.dumps(ui_state, sort_keys=True, default=str)


def ui_fingerprint(state: dict) -> str:
    """Fingerprint of the current inputs; store it next to fresh results."""
    without_results = {k: v for k, v in state.items() if k != "results"}
    return _fingerprint(collect_config(without_results)["ui_state"])


def summarize_results(results) -> dict:
    """Compact P5/P50/P95 + mean summary, small enough to live in the JSON."""
    def stats(arr) -> dict:
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        return {"mean": float(np.mean(arr)), "p5": float(p5),
                "p50": float(p50), "p95": float(p95)}
    return {
        "n_simulations": int(results.n_simulations),
        "enterprise_value": stats(results.total_ev),
        "equity_value": stats(results.equity_values),
        "preis_je_aktie": stats(results.price_per_share),
    }


# ── Import from LLM output ────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_config_text(text: str) -> dict:
    """Extract the config JSON from pasted text (raw JSON or LLM answer).

    Prefers a fenced ```json block; otherwise takes everything from the
    first ``{`` to the last ``}``.  Raises ``ValueError`` if nothing usable
    is found.
    """
    m = _JSON_FENCE.search(text)
    if m:
        raw = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Kein JSON-Block gefunden.")
        raw = text[start:end + 1]
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON fehlerhaft (Zeile {e.lineno}, Spalte {e.colno}): {e.msg}") from e
    if not isinstance(cfg, dict) or ("ui_state" not in cfg and "setup" not in cfg):
        raise ValueError("Ungültiges Format – erwartete 'ui_state' oder Legacy-'setup'.")
    return cfg


# ── Version comparison ────────────────────────────────────────────────

_ACTIVE_SUFFIXES: dict[str, tuple[str, ...]] = {
    DistributionType.FIXED.value:      ("_fixed",),
    DistributionType.NORMAL.value:     ("_n_mu", "_n_sig"),
    DistributionType.LOGNORMAL.value:  ("_ln_mu", "_ln_sig"),
    DistributionType.TRIANGULAR.value: ("_tri_lo", "_tri_mo", "_tri_hi"),
    DistributionType.UNIFORM.value:    ("_uni_lo", "_uni_hi"),
    DistributionType.PERT.value:       ("_pert_lo", "_pert_mo", "_pert_hi"),
}

_SUFFIX_LABELS = {
    "_dtype": "Verteilung", "_fixed": "Wert",
    "_n_mu": "μ", "_n_sig": "σ", "_ln_mu": "Mittelwert", "_ln_sig": "Std.-Abw.",
    "_tri_lo": "Min", "_tri_mo": "Mode", "_tri_hi": "Max",
    "_uni_lo": "Min", "_uni_hi": "Max",
    "_pert_lo": "Min", "_pert_mo": "Mode", "_pert_hi": "Max",
}
_PARAM_LABELS = {
    "rg": "Umsatzwachstum (%)", "em": "EBITDA-Marge (%)", "da": "D&A (% Umsatz)",
    "tx": "Steuersatz (%)", "cx": "CAPEX (% Umsatz)", "nwc": "NWC (% ΔUmsatz)",
    "wacc": "WACC (%)", "tvg": "TV-Wachstum (%)", "evm": "Exit-Multiple",
}
_BRIDGE_LABELS = {
    "bridge_cc": "Holdingkosten p.a. (Mio.)", "bridge_cd": "Diskontierungssatz Holding (%)",
    "bridge_nd": "Nettoverschuldung (Mio.)", "bridge_sh": "Aktien verwässert (Mio.)",
    "bridge_mi": "Minderheitsanteile (Mio.)", "bridge_pn": "Pensionsrückstellungen (Mio.)",
    "bridge_no": "Nicht-operative Assets (Mio.)", "bridge_as": "Beteiligungen (Mio.)",
}
_SEG_FIELD_LABELS = {
    "_name": "Name", "_basrev": "Basisumsatz (Mio.)", "_fyrs": "Prognosejahre",
    "_tv_method": "TV-Methode", "_growth_mode": "Wachstumsmodell",
    "_fade_speed": "Fade-Geschwindigkeit λ", "_param_fade": "Parameter-Fade",
    "_intra_corr": "Intra-Segment-Copula",
}
_SETUP_LABELS = {
    "setup_n_sim": "Iterationen", "setup_seed": "Seed", "setup_n_seg": "Anzahl Segmente",
    "setup_mid_year": "Mid-Year Convention", "setup_ext_bridge": "Erweiterte Bridge",
    "setup_sampling": "Sampling", "setup_corr_enable": "Cross-Segment-Korrelation",
}
_SUFFIXES_BY_LENGTH = sorted(DIST_SUFFIXES, key=len, reverse=True)
_DIST_PREFIX = re.compile(r"^s(\d+)_([a-z]+?)(_term)?$")
_SEG_FIELD = re.compile(r"^seg_(\d+)(_.+)$")
_CORR_KEY = re.compile(r"^corr_(\d+)_(\d+)$")
_INTRA_KEY = re.compile(r"^seg_(\d+)_ic_(\d+)_(\d+)$")


def _split_dist_key(key: str) -> tuple[str, str] | None:
    for sfx in _SUFFIXES_BY_LENGTH:
        if key.endswith(sfx):
            prefix = key[: -len(sfx)]
            if prefix in BRIDGE_PREFIXES or _DIST_PREFIX.match(prefix):
                return prefix, sfx
    return None


def source_key(key: str) -> str:
    """Key under which ``quellen`` documents a field (dist prefix or field key)."""
    split = _split_dist_key(key)
    return split[0] if split else key


def _effective_values(cfg: dict) -> dict:
    """Flatten ui_state, dropping leftovers of inactive distribution types
    and the mirrored half of symmetric correlation matrices."""
    ui = _extract_ui_state(cfg)
    flat: dict = {}
    flat.update(ui.get("setup", {}))
    flat.update({k: v for k, v in ui.get("bridge", {}).items() if k != "setup_ext_bridge"})
    for seg in ui.get("segments", []):
        flat.update(seg)
    flat.update(ui.get("correlation", {}))

    out: dict = {}
    for k, v in flat.items():
        split = _split_dist_key(k)
        if split and split[1] != "_dtype":
            dtype = flat.get(f"{split[0]}_dtype")
            if dtype in _ACTIVE_SUFFIXES and split[1] not in _ACTIVE_SUFFIXES[dtype]:
                continue
        m = _CORR_KEY.match(k) or _INTRA_KEY.match(k)
        if m and int(m.groups()[-2]) >= int(m.groups()[-1]):
            continue
        out[k] = v
    return out


def _label(key: str, values: dict) -> tuple[str, str]:
    """Human-readable (area, parameter) for a flat config key."""
    def seg_area(i: str) -> str:
        name = values.get(f"seg_{i}_name")
        return f"Segment {int(i) + 1}" + (f" · {name}" if name else "")

    split = _split_dist_key(key)
    if split:
        prefix, sfx = split
        if prefix in _BRIDGE_LABELS:
            return "Corporate Bridge", f"{_BRIDGE_LABELS[prefix]} · {_SUFFIX_LABELS[sfx]}"
        i, param, term = _DIST_PREFIX.match(prefix).groups()
        name = _PARAM_LABELS.get(param, param) + (" · Terminal" if term else "")
        return seg_area(i), f"{name} · {_SUFFIX_LABELS[sfx]}"
    if key in _SETUP_LABELS:
        return "Setup", _SETUP_LABELS[key]
    m = _CORR_KEY.match(key)
    if m:
        a, b = (int(g) for g in m.groups())
        return "Korrelation", f"ρ Segment {a + 1} ↔ {b + 1}"
    m = _INTRA_KEY.match(key)
    if m:
        i, r, c = m.groups()
        return seg_area(i), f"ρ {INTRA_PARAM_LABELS[int(r)]} ↔ {INTRA_PARAM_LABELS[int(c)]}"
    m = _SEG_FIELD.match(key)
    if m:
        return seg_area(m.group(1)), _SEG_FIELD_LABELS.get(m.group(2), m.group(2))
    return "Sonstiges", key


def diff_configs(old: dict, new: dict) -> list[dict]:
    """List every input that differs between two saved configs.

    Each row: ``bereich``, ``parameter``, ``alt``, ``neu``, ``quelle``
    (justification from the *new* version's ``quellen`` block, if any).
    """
    a, b = _effective_values(old), _effective_values(new)
    q_old, q_new = old.get("quellen") or {}, new.get("quellen") or {}
    rows = []
    for key in dict.fromkeys([*a, *b]):  # keep form order (Min, Mode, Max)
        va, vb = a.get(key), b.get(key)
        if va == vb:
            continue
        bereich, parameter = _label(key, b if key in b else a)
        sk = source_key(key)
        quelle = q_new.get(sk, "")
        if quelle and quelle == q_old.get(sk):
            # Value moved but justification did not – likely a manual edit.
            quelle = f"⚠️ Begründung unverändert: {quelle}"
        rows.append({
            "bereich": bereich,
            "parameter": parameter,
            "alt": "–" if va is None else va,
            "neu": "–" if vb is None else vb,
            "quelle": quelle,
        })
    return rows


# ── State → SimulationConfig (single conversion point) ────────────────
#
# The Streamlit pages render widgets whose values live in the session
# state under the keys above.  These functions turn such a flat state
# (``st.session_state`` or ``apply_config(cfg, {})``) into domain objects,
# so the app and headless runs share one conversion: % → decimal,
# Lognormal display mean/std → underlying μ/σ, and the segment rules.
# Defaults mirror the widget defaults for keys that are absent.

def dist_config_from_state(
    state: dict, prefix: str, default: float = 0.0, is_percentage: bool = False,
):
    """DistributionConfig for one ``<prefix>_*`` input (display units in state)."""
    from domain.models import DistributionConfig

    div = 100.0 if is_percentage else 1.0
    d = default

    def g(sfx: str, dflt: float) -> float:
        return float(state.get(f"{prefix}{sfx}", dflt))

    dist_type = DistributionType(state.get(f"{prefix}_dtype", DistributionType.FIXED.value))
    c = DistributionConfig(dist_type=dist_type)

    if dist_type == DistributionType.FIXED:
        c.fixed_value = g("_fixed", d) / div
    elif dist_type == DistributionType.NORMAL:
        c.mean = g("_n_mu", d) / div
        c.std = g("_n_sig", max(abs(d) * 0.2, 0.01)) / div
    elif dist_type == DistributionType.LOGNORMAL:
        mean = g("_ln_mu", max(d, 0.01)) / div
        std = g("_ln_sig", max(abs(d) * 0.2, 0.01)) / div
        if mean > 1e-12 and std > 1e-12:
            sigma_sq = float(np.log(1.0 + (std / mean) ** 2))
            c.ln_sigma = float(np.sqrt(sigma_sq))
            c.ln_mu = float(np.log(mean) - sigma_sq / 2.0)
        else:
            c.ln_mu = float(np.log(max(mean, 1e-12)))
            c.ln_sigma = 0.1
    elif dist_type == DistributionType.TRIANGULAR:
        c.low = g("_tri_lo", d * 0.8) / div
        c.mode = g("_tri_mo", d) / div
        c.high = g("_tri_hi", d * 1.2) / div
    elif dist_type == DistributionType.UNIFORM:
        c.low = g("_uni_lo", d * 0.8) / div
        c.high = g("_uni_hi", d * 1.2) / div
    elif dist_type == DistributionType.PERT:
        c.low = g("_pert_lo", d * 0.7) / div
        c.mode = g("_pert_mo", d) / div
        c.high = g("_pert_hi", d * 1.3) / div
    return c


def _psd_or_none(matrix: list[list[float]]) -> list[list[float]] | None:
    """Return the matrix if positive semi-definite, else None (as the UI does)."""
    return None if np.any(np.linalg.eigvalsh(np.array(matrix)) < -1e-8) else matrix


def segment_config_from_state(state: dict, i: int):
    """SegmentConfig for segment *i* – same rules as the segments page."""
    from domain.models import (
        DEFAULT_INTRA_PARAM_CORR, DistributionConfig, RevenueGrowthMode,
        SegmentConfig, TerminalValueMethod,
    )

    def dist(param: str, default: float, pct: bool = True):
        return dist_config_from_state(state, f"s{i}_{param}", default, pct)

    growth_mode = RevenueGrowthMode(
        state.get(f"seg_{i}_growth_mode", RevenueGrowthMode.CONSTANT.value))
    is_fade = growth_mode == RevenueGrowthMode.FADE
    fade_speed = float(state.get(f"seg_{i}_fade_speed", 0.5)) if is_fade else 0.5

    # Terminal targets only exist when fade AND parameter fade are on.
    terms = dict.fromkeys(["em", "da", "tx", "cx", "nwc"])
    if is_fade and bool(state.get(f"seg_{i}_param_fade", False)):
        for p, dflt in [("em", 18.0), ("da", 3.0), ("tx", 25.0), ("cx", 4.0), ("nwc", 8.0)]:
            terms[p] = dist(f"{p}_term", dflt)

    tv_method = TerminalValueMethod(
        state.get(f"seg_{i}_tv_method", TerminalValueMethod.GORDON_GROWTH.value))
    if tv_method == TerminalValueMethod.GORDON_GROWTH:
        tv_growth = dist("tvg", 2.0)
        tv_multiple = DistributionConfig(fixed_value=10.0)
    else:
        tv_multiple = dist("evm", 10.0, pct=False)
        tv_growth = dist("tvg", 2.0) if is_fade else DistributionConfig(fixed_value=0.02)

    intra = None
    if bool(state.get(f"seg_{i}_intra_corr", False)):
        m = [list(r) for r in DEFAULT_INTRA_PARAM_CORR]
        for r in range(7):
            for c in range(r + 1, 7):
                v = float(state.get(f"seg_{i}_ic_{r}_{c}", m[r][c]))
                m[r][c] = m[c][r] = v
        intra = _psd_or_none(m)

    return SegmentConfig(
        name=state.get(f"seg_{i}_name", f"Segment {i + 1}"),
        base_revenue=float(state.get(f"seg_{i}_basrev", 1_000.0)),
        forecast_years=int(state.get(f"seg_{i}_fyrs", 5)),
        revenue_growth=dist("rg", 5.0),
        ebitda_margin=dist("em", 20.0),
        da_pct_revenue=dist("da", 3.0),
        tax_rate=dist("tx", 25.0),
        capex_pct_revenue=dist("cx", 5.0),
        nwc_pct_delta_revenue=dist("nwc", 10.0),
        wacc=dist("wacc", 9.0),
        terminal_method=tv_method,
        terminal_growth_rate=tv_growth,
        exit_multiple=tv_multiple,
        revenue_growth_mode=growth_mode,
        fade_speed=fade_speed,
        ebitda_margin_terminal=terms["em"],
        da_pct_revenue_terminal=terms["da"],
        tax_rate_terminal=terms["tx"],
        capex_pct_revenue_terminal=terms["cx"],
        nwc_pct_delta_revenue_terminal=terms["nwc"],
        intra_param_correlation=intra,
    )


def bridge_dists_from_state(state: dict) -> dict:
    """Bridge DistributionConfigs keyed by prefix; extended items None when off."""
    core = {"bridge_cc": (50.0, False), "bridge_cd": (9.0, True),
            "bridge_nd": (500.0, False), "bridge_sh": (100.0, False)}
    out = {p: dist_config_from_state(state, p, d, pct) for p, (d, pct) in core.items()}
    ext = bool(state.get("setup_ext_bridge", False))
    for p in ("bridge_mi", "bridge_pn", "bridge_no", "bridge_as"):
        out[p] = dist_config_from_state(state, p, 0.0) if ext else None
    return out


def split_bridge_param(dist, fallback: float = 0.0):
    """(scalar, stochastic override): "Fest" gives the scalar only, else both."""
    if dist is None:
        return fallback, None
    if dist.dist_type == DistributionType.FIXED:
        return dist.fixed_value, None
    return dist.representative_value(), dist


_BRIDGE_STOCHASTIC_FIELD = {
    "annual_corporate_costs": "stochastic_corporate_costs",
    "corporate_cost_discount_rate": "stochastic_corporate_cost_discount_rate",
    "net_debt": "stochastic_net_debt",
    "shares_outstanding": "stochastic_shares",
    "minority_interests": "stochastic_minority_interests",
    "pension_liabilities": "stochastic_pension_liabilities",
    "non_operating_assets": "stochastic_non_operating_assets",
    "associate_investments": "stochastic_associate_investments",
}
_BRIDGE_FALLBACK = {"bridge_cc": 50.0, "bridge_cd": 0.09, "bridge_nd": 500.0, "bridge_sh": 100.0}


def corporate_bridge_from_dists(dists: dict) -> CorporateBridgeConfig:
    """CorporateBridgeConfig from prefix → DistributionConfig | None."""
    kw = {}
    for field, prefix in _BRIDGE_FIELD_TO_PREFIX.items():
        val, stoch = split_bridge_param(dists.get(prefix), _BRIDGE_FALLBACK.get(prefix, 0.0))
        kw[field], kw[_BRIDGE_STOCHASTIC_FIELD[field]] = val, stoch
    return CorporateBridgeConfig(**kw)


def segment_correlation_from_state(state: dict, n_seg: int) -> list[list[float]] | None:
    """Cross-segment matrix – only when enabled, at least 2 segments and PSD."""
    if not bool(state.get("setup_corr_enable", False)) or n_seg < 2:
        return None
    m = [[1.0] * n_seg for _ in range(n_seg)]
    for r in range(n_seg):
        for c in range(r + 1, n_seg):
            m[r][c] = m[c][r] = float(state.get(f"corr_{r}_{c}", 0.3))
    return _psd_or_none(m)


def simulation_config_from_state(state: dict):
    """Complete SimulationConfig from a flat (session-)state mapping."""
    from domain.models import SamplingMethod, SimulationConfig

    n_seg = int(state.get("setup_n_seg", 2))
    return SimulationConfig(
        n_simulations=int(state.get("setup_n_sim", 10_000)),
        random_seed=int(state.get("setup_seed", 42)),
        segments=[segment_config_from_state(state, i) for i in range(n_seg)],
        corporate_bridge=corporate_bridge_from_dists(bridge_dists_from_state(state)),
        mid_year_convention=bool(state.get("setup_mid_year", True)),
        sampling_method=SamplingMethod(
            state.get("setup_sampling", SamplingMethod.PSEUDO_RANDOM.value)),
        segment_correlation=segment_correlation_from_state(state, n_seg),
    )


def build_simulation_config(cfg: dict):
    """SimulationConfig from a saved config dict (v2 or legacy v1)."""
    return simulation_config_from_state(apply_config(cfg, {}))
