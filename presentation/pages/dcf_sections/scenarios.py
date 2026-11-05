"""Scenarios and reverse DCF tab (N1/N2) – UI only; logic in application.scenario_service."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from application.scenario_service import (
    PRESETS, group_scenarios, preset_patterns, reverse_dcf, run_scenarios,
)
from infrastructure.config_io import (
    SCENARIO_RESULTS_KEY, apply_config, collect_config, ui_fingerprint,
)

_HELP = """
**Muster** ist ein Feldname mit Platzhalter `*`, z. B. `s*_wacc` (WACC aller Segmente), `s0_em` und
`s0_em_term` (Marge Segment 1, Start und terminal), `bridge_cd`, `seg_0_basrev`.
**Op** `+` verschiebt alle Lagewerte der Verteilung (Min/Modus/Max bzw. μ, Streuung bleibt),
`=` setzt den Modus und verschiebt Min/Max mit. Ein exakter Feldname wie `bridge_mi_fixed` setzt genau
dieses Feld. **Wert** in Anzeige-Einheiten (Prozentpunkte, Mio.). Mehrere Zeilen mit demselben
Szenario-Namen bilden ein Szenario. Gerechnet wird mit Seed und Iterationen der Basis, ein Δ enthält
dann kein Stichprobenrauschen. Ausnahme: Ändert ein Szenario einen Verteilungstyp, ist die Zeile markiert.
"""


def _model_state() -> dict:
    """Flat state exactly as saving and reloading would give it (one conversion path)."""
    cfg = collect_config({k: v for k, v in st.session_state.items() if k != "results"})
    return {k: v for k, v in apply_config(cfg, {}).items() if k != "cfg_passthrough"}


def _store(**parts) -> None:
    fp = ui_fingerprint(dict(st.session_state))
    defs = st.session_state.setdefault("cfg_passthrough", {}).get("szenarien")
    cur = st.session_state.get(SCENARIO_RESULTS_KEY) or {}
    if cur.get("fingerprint") != fp or cur.get("definitionen") != defs:
        cur = {"fingerprint": fp, "definitionen": defs}
    cur.update(parts)
    st.session_state[SCENARIO_RESULTS_KEY] = cur


def _current() -> dict:
    cur = st.session_state.get(SCENARIO_RESULTS_KEY) or {}
    ok = (cur.get("fingerprint") == ui_fingerprint(dict(st.session_state))
          and cur.get("definitionen") == st.session_state.get("cfg_passthrough", {}).get("szenarien"))
    return cur if ok else {}


def render_scenarios_section() -> None:
    passthrough = st.session_state.setdefault("cfg_passthrough", {})
    meta = passthrough.setdefault("metadata", {})

    st.subheader("🎯 Szenarien")
    with st.expander("ℹ️ Wie schreibe ich ein Szenario?", expanded=False):
        st.markdown(_HELP)
    # The editor gets a stable input, built once per loaded definition set (never its own
    # output), and definitions are written back only when an edit changed them.
    loaded = passthrough.get("szenarien")
    if st.session_state.get("_scen_seen") != loaded or "_scen_init" not in st.session_state:
        st.session_state["_scen_init"] = pd.DataFrame(
            [{"szenario": s["name"], "muster": o["muster"], "op": o.get("op", "="), "wert": str(o["wert"])}
             for s in loaded or [] for o in s["ops"]], columns=["szenario", "muster", "op", "wert"])
        st.session_state["_scen_seen"] = loaded
        st.session_state.pop("scen_editor", None)
    st.caption("Beispiel: Szenario „WACC +1 pp“, Muster `s*_wacc`, Op `+`, Wert `1` (und eine zweite Zeile "
               "mit `bridge_cd` für die Holding).")
    edited = st.data_editor(
        st.session_state["_scen_init"], num_rows="dynamic", width="stretch", key="scen_editor",
        hide_index=True,
        column_config={
            "szenario": st.column_config.TextColumn("Szenario"),
            "muster": st.column_config.TextColumn("Muster"),
            "op": st.column_config.SelectboxColumn("Op", options=["+", "="]),
            "wert": st.column_config.TextColumn("Wert"),
        },
    )
    defs = group_scenarios(edited.to_dict("records")) or None
    if defs != group_scenarios(st.session_state["_scen_init"].to_dict("records")):
        if defs is None:
            passthrough.pop("szenarien", None)
        else:
            passthrough["szenarien"] = defs
        st.session_state["_scen_seen"] = passthrough.get("szenarien")

    if st.button("▶ Szenarien rechnen", type="primary", disabled="szenarien" not in passthrough):
        bar = st.progress(0.0, text="Szenarien …")
        res = run_scenarios(_model_state(), passthrough["szenarien"],
                            progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        _store(szenarien=res)
    res = _current().get("szenarien")
    if res:
        b = res["basis"]
        st.caption(f"Basis P50 {b['p50']:,.2f} (P5 {b['p5']:,.2f} / P95 {b['p95']:,.2f})")
        st.dataframe([{
            "Szenario": r["name"],
            "P50": None if r["p50"] is None else round(r["p50"], 2),
            "Δ": None if r["delta"] is None else round(r["delta"], 2),
            "P5": None if r["p5"] is None else round(r["p5"], 2),
            "P95": None if r["p95"] is None else round(r["p95"], 2),
            "Hinweis": "; ".join(filter(None, [r["fehler"], "Δ enthält Stichprobenrauschen" if r["rauschen"] else "",
                                                *r["warnungen"]])),
        } for r in res["zeilen"]], hide_index=True, width="stretch")

    st.divider()
    st.subheader("🔁 Reverse DCF: Was der Kurs voraussetzt")
    state = _model_state()
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    preset = c1.selectbox("Hebel", [*PRESETS, "Freies Muster"], key="rdcf_preset")
    default_lo, default_hi = (PRESETS[preset][1:] if preset in PRESETS else (-5.0, 5.0))
    pats_txt = st.text_input("Muster (kommagetrennt, gleiche Verschiebung auf alle)",
                             value=", ".join(preset_patterns(preset, state)) if preset in PRESETS else "",
                             key=f"rdcf_pats_{preset}")
    lo = c2.number_input("Δ von", value=float(default_lo), step=0.5, key=f"rdcf_lo_{preset}")
    hi = c3.number_input("Δ bis", value=float(default_hi), step=0.5, key=f"rdcf_hi_{preset}")
    kurs = c4.number_input("Kurs", value=float(meta.get("kurs_bei_recherche") or 0.0), key="rdcf_kurs")
    pats = [p.strip() for p in pats_txt.split(",") if p.strip()]
    if st.button("▶ Reverse DCF rechnen", disabled=not pats or not kurs):
        bar = st.progress(0.0, text="Bisektion …")
        r = reverse_dcf(state, pats, kurs, lo, hi, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        _store(reverse_dcf={**r, "hebel": preset})
    r = _current().get("reverse_dcf")
    if r:
        if r["status"] == "ok":
            st.success(f"P50 = Kurs {r['ziel']:,.2f} bei Verschiebung **{r['delta']:+.2f}** auf {', '.join(r['muster'])}")
            st.dataframe([{"Feld": k, "impliziter Wert": round(v, 2)} for k, v in r["implizit"].items()],
                         hide_index=True)
        else:
            st.warning(f"Im Bereich nicht erreichbar: {r['detail']}")
