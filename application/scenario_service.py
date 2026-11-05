"""
Scenarios, reverse DCF and version walk – one primitive for all three.

An *op list* changes a flat config state (as produced by ``apply_config``):
``{"muster": "s*_wacc", "op": "+", "wert": 1.0}``. Values are display units
(pp, Mio.). A pattern that names a distribution prefix moves every location
value of the active distribution type (spreads stay); a pattern that names a
plain key changes that key. All runs go through ``simulation_config_from_state``
(single conversion point, A1) with the base seed and iteration count, so a Δ
contains no sampling noise unless an op changes a distribution type or the
segment count (flagged as ``rauschen``).
"""
from __future__ import annotations

import copy
import fnmatch
import re
from typing import Callable

import numpy as np

from domain.checks import OK, config_checks
from domain.models import DistributionType, TerminalValueMethod
from infrastructure.config_io import (
    _ACTIVE_SUFFIXES, BRIDGE_PREFIXES, DIST_PARAMS, SETUP_KEYS, _split_dist_key, apply_config,
    simulation_config_from_state,
)
from infrastructure.monte_carlo_engine import MonteCarloEngine

# Location values per distribution type; the first one is the "centre" for "=".
_LOCATION = {
    DistributionType.FIXED.value: ("_fixed",),
    DistributionType.NORMAL.value: ("_n_mu",),
    DistributionType.LOGNORMAL.value: ("_ln_mu",),
    DistributionType.TRIANGULAR.value: ("_tri_mo", "_tri_lo", "_tri_hi"),
    DistributionType.PERT.value: ("_pert_mo", "_pert_lo", "_pert_hi"),
    DistributionType.UNIFORM.value: ("_uni_lo", "_uni_hi"),
}
_ORDERED = {DistributionType.TRIANGULAR.value: ("_tri_lo", "_tri_mo", "_tri_hi"),
            DistributionType.PERT.value: ("_pert_lo", "_pert_mo", "_pert_hi"),
            DistributionType.UNIFORM.value: ("_uni_lo", "_uni_hi")}

Progress = Callable[[float, str], None] | None


def _dist_prefixes(state: dict) -> list[str]:
    return [k[:-6] for k in state if k.endswith("_dtype")]


_SEG_FIELD_KEY = re.compile(r"^seg_\d+_(name|basrev|fyrs|tv_method|growth_mode|fade_speed|param_fade|"
                            r"intra_corr|anteil|ic_\d_\d)$")


def _is_model_key(key: str) -> bool:
    """True for keys the model reads (so creating them is not a typo)."""
    if key in SETUP_KEYS or _SEG_FIELD_KEY.match(key):
        return True
    split = _split_dist_key(key)
    if not split:
        return False
    prefix = split[0]
    return prefix in BRIDGE_PREFIXES or prefix.split("_", 1)[1] in DIST_PARAMS


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def apply_ops(state: dict, ops: list[dict]) -> tuple[dict, bool]:
    """Return (changed copy of *state*, noise flag). Raises ValueError on bad ops/results."""
    s = copy.deepcopy(state)
    noise = False
    for o in ops:
        pat, op, val = str(o["muster"]).strip(), str(o.get("op", "=")).strip(), _num(o["wert"])
        if op not in ("=", "+"):
            raise ValueError(f"Operation „{op}“ unbekannt (erlaubt: = oder +)")
        prefixes = [p for p in _dist_prefixes(s) if fnmatch.fnmatchcase(p, pat)]
        keys = [k for k in s if fnmatch.fnmatchcase(k, pat)]
        if not prefixes and not keys and op == "=" and not any(c in pat for c in "*?["):
            # a valid model key that is not in the state yet, e.g. "bridge_mi_fixed" after a
            # dtype change or "bridge_mi_dtype" in an old file; typos still fail
            if _is_model_key(pat):
                keys = [pat]
        if not prefixes and not keys:
            raise ValueError(f"Muster „{pat}“ trifft kein Feld")
        if prefixes:
            if op == "+" and isinstance(val, str):
                raise ValueError(f"„{pat} + {val}“: Verschieben braucht eine Zahl")
            for p in prefixes:
                sfx = _LOCATION.get(s[f"{p}_dtype"], ())
                present = [x for x in sfx if f"{p}{x}" in s]
                if not present:
                    raise ValueError(f"„{p}“ hat keine Lagewerte")
                if op == "+":
                    delta = val
                elif s[f"{p}_dtype"] == DistributionType.UNIFORM.value:
                    delta = val - (float(s[f"{p}_uni_lo"]) + float(s[f"{p}_uni_hi"])) / 2.0
                else:  # "=": centre to the value, spread unchanged
                    delta = val - float(s[f"{p}{present[0]}"])
                for x in present:
                    s[f"{p}{x}"] = float(s[f"{p}{x}"]) + delta
        else:
            for k in keys:
                if op == "+":
                    s[k] = float(s[k]) + val
                else:
                    noise |= k.endswith("_dtype") and s.get(k) != val or k == "setup_n_seg"
                    s[k] = val
    for p in _dist_prefixes(s):
        order = _ORDERED.get(s[f"{p}_dtype"])
        if order and all(f"{p}{x}" in s for x in order):
            v = [float(s[f"{p}{x}"]) for x in order]
            if v != sorted(v):
                raise ValueError(f"„{p}“: Min ≤ Modus ≤ Max verletzt ({', '.join(f'{x:g}' for x in v)})")
    return s, noise


def run_state(state: dict, n: int | None = None, seed: int | None = None):
    st = dict(state)
    if n is not None:
        st["setup_n_sim"] = n
    if seed is not None:
        st["setup_seed"] = seed
    sc = simulation_config_from_state(st)
    return sc, MonteCarloEngine(sc).run()


def _p(r) -> tuple[float, float, float]:
    p5, p50, p95 = np.percentile(r.price_per_share, [5, 50, 95])
    return float(p5), float(p50), float(p95)


def _warnings(sc) -> list[str]:
    return [f"C{c.id} {c.name}" for c in config_checks(sc.segments) if c.status != OK]


def group_scenarios(rows: list[dict]) -> list[dict]:
    """Editor rows (szenario, muster, op, wert) → [{"name", "ops"}], order kept."""
    out: dict[str, list] = {}
    for r in rows:
        name = str(r.get("szenario") or "").strip()
        if name and str(r.get("muster") or "").strip():
            out.setdefault(name, []).append({"muster": str(r["muster"]).strip(), "op": r.get("op") or "=",
                                             "wert": _num(r["wert"])})
    return [{"name": k, "ops": v} for k, v in out.items()]


def run_scenarios(state: dict, scenarios: list[dict], progress: Progress = None) -> dict:
    """Base + every scenario against the base (same seed and n)."""
    sc, r = run_state(state)
    base = _p(r)
    rows = []
    for i, scen in enumerate(scenarios):
        if progress:
            progress(i / max(len(scenarios), 1), scen["name"])
        try:
            st, noise = apply_ops(state, scen["ops"])
            sc_i, r_i = run_state(st)
            p5, p50, p95 = _p(r_i)
            rows.append({"name": scen["name"], "p5": p5, "p50": p50, "p95": p95, "delta": p50 - base[1],
                         "rauschen": noise, "warnungen": _warnings(sc_i), "fehler": ""})
        except ValueError as e:
            rows.append({"name": scen["name"], "p5": None, "p50": None, "p95": None, "delta": None,
                         "rauschen": False, "warnungen": [], "fehler": str(e)})
    if progress:
        progress(1.0, "fertig")
    return {"basis": {"p5": base[0], "p50": base[1], "p95": base[2]}, "zeilen": rows}


# ── Reverse DCF ───────────────────────────────────────────────────────

PRESETS = {
    "WACC alle Segmente + Holding": (["s*_wacc", "bridge_cd"], 0.0, 6.0),
    "TV-g der Gordon-Segmente": (None, -6.0, 0.0),  # patterns depend on the config
}


def preset_patterns(name: str, state: dict) -> list[str]:
    pats = PRESETS[name][0]
    if pats is not None:
        return pats
    n = int(state.get("setup_n_seg", 0))
    return [f"s{i}_tvg" for i in range(n)
            if state.get(f"seg_{i}_tv_method", TerminalValueMethod.GORDON_GROWTH.value)
            == TerminalValueMethod.GORDON_GROWTH.value]


def _shift_bounds(state: dict, patterns: list[str]) -> tuple[float | None, float | None]:
    """(min Δ, max Δ) that keep every Gordon segment's TV-g ≥ 0.5 pp below its WACC.

    A shift on TV-g caps Δ from above, a shift on WACC caps it from below; shifting
    both by the same Δ keeps the gap and sets no bound.
    """
    sc = simulation_config_from_state(state)
    lo = hi = None
    for i, seg in enumerate(sc.segments):
        if seg.terminal_method != TerminalValueMethod.GORDON_GROWTH:
            continue
        hits = {p: any(fnmatch.fnmatchcase(f"s{i}_{p}", m) for m in patterns) for p in ("tvg", "wacc")}
        if hits["tvg"] == hits["wacc"]:
            continue
        sfx = _LOCATION.get(state.get(f"s{i}_tvg_dtype"), ())
        g_hi = max(float(state[f"s{i}_tvg{x}"]) for x in sfx if f"s{i}_tvg{x}" in state)
        room = float(seg.wacc.representative_value()) * 100 - 0.5 - g_hi
        if hits["tvg"]:
            hi = room if hi is None else min(hi, room)
        else:
            lo = -room if lo is None else max(lo, -room)
    return lo, hi


def reverse_dcf(state: dict, patterns: list[str], target: float, lo: float, hi: float,
                tol: float = 0.005, progress: Progress = None) -> dict:
    """Shift Δ (same Δ on every pattern) at which the P50 equals *target*."""
    b_lo, b_hi = _shift_bounds(state, patterns)
    if b_hi is not None and hi > b_hi:
        hi = b_hi
    if b_lo is not None and lo < b_lo:
        lo = b_lo
    if hi <= lo:
        return {"status": "nicht erreichbar", "detail": "Kein Suchbereich, TV-g liegt schon nahe am WACC"}

    def p50(d):
        st, _ = apply_ops(state, [{"muster": p, "op": "+", "wert": d} for p in patterns])
        return _p(run_state(st)[1])[1]

    f_lo, f_hi = p50(lo) - target, p50(hi) - target
    if f_lo * f_hi > 0:
        return {"status": "nicht erreichbar",
                "detail": f"P50 liegt bei Δ {lo:+.2f} und {hi:+.2f} auf derselben Seite des Kurses "
                          f"({f_lo + target:,.2f} / {f_hi + target:,.2f})"}
    steps = int(np.ceil(np.log2(max((hi - lo) / tol, 2))))
    for k in range(steps):
        if progress:
            progress(k / steps, f"Bisektion {k + 1}/{steps}")
        mid = (lo + hi) / 2.0
        f_mid = p50(mid) - target
        if f_mid * f_lo > 0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    d = (lo + hi) / 2.0
    st, _ = apply_ops(state, [{"muster": p, "op": "+", "wert": d} for p in patterns])
    implied = {}
    for p in patterns:
        for q in [x for x in _dist_prefixes(st) if fnmatch.fnmatchcase(x, p)]:
            c = _LOCATION[st[f"{q}_dtype"]][0]
            name = st.get(f"seg_{q[1:].split('_')[0]}_name", q) if re.match(r"s\d+_", q) else q
            implied[f"{name} · {q.split('_', 1)[1]}"] = float(st[f"{q}{c}"])
    if progress:
        progress(1.0, "fertig")
    return {"status": "ok", "delta": d, "muster": patterns, "ziel": target, "implizit": implied}


# ── Version walk ──────────────────────────────────────────────────────

def _area(key: str) -> str | None:
    if key.startswith("bridge_") or key == "setup_ext_bridge":
        return "Bridge"
    m = re.match(r"^(?:seg_(\d+)_|s(\d+)_)", key)
    if m:
        return f"Segment {int(m.group(1) or m.group(2)) + 1}"
    if key.startswith("corr_") or key == "setup_corr_enable":
        return "Korrelation"
    if key in ("setup_mid_year", "setup_sampling"):
        return "Setup"  # method choices, shown as their own step
    return None  # iterations, seed, segment count: not assumptions


_SETUP_DEFAULTS = {"setup_mid_year": True, "setup_sampling": "Pseudo-Random (Standard)",
                   "setup_ext_bridge": False, "setup_corr_enable": False}


def _effective(state: dict) -> dict:
    """Values the model reads: inactive distribution leftovers dropped, setup defaults filled."""
    out = {**_SETUP_DEFAULTS, **state}
    for k in list(out):
        split = _split_dist_key(k)
        if split and split[1] != "_dtype":
            active = _ACTIVE_SUFFIXES.get(out.get(f"{split[0]}_dtype"))
            if active is not None and split[1] not in active:
                del out[k]
    return out


def _state_of(cfg: dict) -> dict:
    return {k: v for k, v in apply_config(cfg, {}).items() if k != "cfg_passthrough"}


def version_date(cfg: dict) -> str:
    """Stichtag of a saved config (metadata), else the save date."""
    return str((cfg.get("metadata") or {}).get("stichtag") or str(cfg.get("saved_at", ""))[:10])


def named_steps_for(cfg_a: dict, cfg_b: dict) -> list[dict]:
    """Named steps in B's ``ueberleitung`` block that start from A (``von`` = A's date, or unset)."""
    d = version_date(cfg_a)
    return [s for s in cfg_b.get("ueberleitung") or [] if s.get("von") in (None, "", d)]


def walk(cfg_a: dict, cfg_b: dict, named_steps: list[dict] | None = None,
         progress: Progress = None) -> dict:
    """A → B in steps: engine effect, named op lists, then remaining differences by area."""
    a, b = _state_of(cfg_a), _state_of(cfg_b)
    n, seed = int(b.get("setup_n_sim", 10_000)), int(b.get("setup_seed", 42))
    steps: list[tuple[str, dict]] = []
    cur = a
    for s in named_steps or []:
        cur, _ = apply_ops(cur, s["ops"])
        steps.append((s["name"], cur))
    if int(cur.get("setup_n_seg", 0)) != int(b.get("setup_n_seg", 0)):
        areas = ["Struktur"]
    else:
        ea, eb = _effective(cur), _effective(b)
        areas = list(dict.fromkeys(_area(k) for k in [*ea, *eb]
                                   if _area(k) and ea.get(k) != eb.get(k)))
        areas.sort(key=lambda x: (x != "Bridge", x == "Korrelation", x))
    for area in areas:
        nxt = {k: v for k, v in cur.items() if not (area == "Struktur" and _area(k) or _area(k) == area)}
        nxt.update({k: v for k, v in b.items() if area == "Struktur" and _area(k) or _area(k) == area})
        if area == "Struktur":
            nxt["setup_n_seg"] = b.get("setup_n_seg")
        cur = nxt
        steps.append((area, cur))

    snap = (cfg_a.get("ergebnis") or {}).get("preis_je_aktie", {}).get("p50")
    rows = []
    total = len(steps) + 1
    if progress:
        progress(0.0, "Version A")
    p_a = _p(run_state(a, n, seed)[1])[1]
    rows.append({"schritt": "A auf heutigem Modell" + ("" if snap is None else f" (gespeichert {snap:,.2f})"),
                 "p50": p_a, "delta": None if snap is None else p_a - snap})
    prev = p_a
    for i, (label, st) in enumerate(steps, 1):
        if progress:
            progress(i / total, label)
        p = _p(run_state(st, n, seed)[1])[1]
        rows.append({"schritt": label, "p50": p, "delta": p - prev})
        prev = p
    p_b = _p(run_state(b, n, seed)[1])[1]
    if progress:
        progress(1.0, "fertig")
    return {"zeilen": rows, "p50_b": p_b, "schliesst": abs(prev - p_b) < 1e-9}

