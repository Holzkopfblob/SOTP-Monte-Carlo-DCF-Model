"""Research-Kit: gemeinsame Bausteine für SOTP-Analysen und Aktualisierungen.

Ersetzt die in jedem Analyse-Ordner kopierten Helfer (d, segment, _shift, state_of, to_cfg, ui_check …).
Vorlage war Documents/TELKO-SOTP-2026-11/telko_common.py. Firmenskripte enthalten nur noch Fakten und Urteil:

  Erstanalyse:      cfg = build_config(...);  cfg = evaluate(cfg, kurs);  save(cfg, ...)
  Quartals-Update:  steps = [...] + [roll_quarter(old, ...)];  cfg = update_config(old, steps, ...);  evaluate(...)

Muster „Quartal rollen“ (NVIDIA 18.11.2026): Rollt die LTM-Basis ein Quartal weiter, zählt ein unverändertes
Jahr-1-Wachstum das schon eingetretene Quartal doppelt. roll_quarter verschiebt deshalb den Fade-Pfad um ¼ Jahr
(g'(t) = g(t + ¼)); was das Quartal Neues gezeigt hat, steckt allein in der neuen Basis (plan_check zeigt es).
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from application.scenario_service import (
    apply_ops, named_steps_for, preset_patterns, reverse_dcf, run_scenarios, walk,
)
from domain.checks import result_checks
from domain.models import DEFAULT_INTRA_PARAM_CORR
from infrastructure.config_io import apply_config, build_simulation_config, parse_config_text, summarize_results
from infrastructure.monte_carlo_engine import MonteCarloEngine

VAULT = Path(os.environ.get("SOTP_BEWERTUNGEN",
                            r"C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen\SOTP-Monte-Carlo-DCF_Bewertungen"))
FEST, PERT, NORM, TRI = ("Fest (Deterministisch)", "PERT-Verteilung", "Normalverteilung", "Dreiecksverteilung")
FADE = "Fade-Modell (g konvergiert zum Terminal-Wachstum)"
GORDON = "Gordon Growth Model (Ewige Rente)"
IC = [list(r) for r in DEFAULT_INTRA_PARAM_CORR]
ERP = 4.23  # Damodaran Mature-Market-ERP, Stand 05.01.2026 [primär geprüft]


# ── Bausteine der Konfiguration ──────────────────────────────────────────────────────────────────────────

def d(p: str, kind: str, *v) -> dict:
    """Verteilungsfelder eines Parameters, z. B. d("s0_rg", PERT, lo, mo, hi)."""
    keys = {FEST: ["fixed"], PERT: ["pert_lo", "pert_mo", "pert_hi"],
            NORM: ["n_mu", "n_sig"], TRI: ["tri_lo", "tri_mo", "tri_hi"]}[kind]
    return {f"{p}_dtype": kind, **{f"{p}_{k}": x for k, x in zip(keys, v)}}


def segment(i, name, rev, yrs, lam, rg, em, em_t, da, da_t, tx, tx_t, cx, cx_t, nwc, nwc_t, wacc_, tvg) -> dict:
    """Ein Segment im Fade-Modell mit Parameter-Fade und Gordon-Endwert; Parameter als (kind, *werte)."""
    s = {f"seg_{i}_name": name, f"seg_{i}_basrev": round(rev, 0), f"seg_{i}_fyrs": yrs,
         f"seg_{i}_growth_mode": FADE, f"seg_{i}_fade_speed": lam, f"seg_{i}_param_fade": True,
         f"seg_{i}_tv_method": GORDON, f"seg_{i}_intra_corr": True}
    for p, v in [("rg", rg), ("em", em), ("em_term", em_t), ("da", da), ("da_term", da_t),
                 ("tx", tx), ("tx_term", tx_t), ("cx", cx), ("cx_term", cx_t),
                 ("nwc", nwc), ("nwc_term", nwc_t), ("wacc", wacc_), ("tvg", tvg)]:
        s.update(d(f"s{i}_{p}", *v))
    for r in range(7):
        for c in range(7):
            s[f"seg_{i}_ic_{r}_{c}"] = IC[r][c]
    return s


def fade_start(year1: float, term: float, lam: float) -> float:
    """Fade-Startwert, mit dem Jahr 1 (= term + (start − term)·e^−λ) den Zielwert trifft."""
    return term + (year1 - term) / math.exp(-lam)


def fade_pert(m1: float, mt: float, lam: float, half: float) -> tuple:
    """PERT für den Fade-STARTWERT: Jahr 1 hat Modus m1 und ein P5–P95-Band von ±half (Regel 14).
    Spannweite = 1,6 × gewünschte P5–P95-Breite des Starts (Faustregel 2.3)."""
    k = math.exp(-lam)
    mo = mt + (m1 - mt) / k
    span = 1.6 * 2 * half / k
    return (PERT, round(mo - span / 2, 2), round(mo, 2), round(mo + span / 2, 2))


def wacc(rf: float, d_v: float, spread: float, tax: float, beta_u: float, beta_src: str,
         crp: float = 0.0, erp: float = ERP) -> tuple[float, str]:
    """WACC und Herleitungstext: β levered nach Hamada, k_e = rf + β·ERP + CRP, k_d = (rf + Spread)·(1−t)."""
    d_e = d_v / (1 - d_v)
    beta = beta_u * (1 + (1 - tax / 100) * d_e)
    ke, kd = rf + beta * erp + crp, (rf + spread) * (1 - tax / 100)
    w = (1 - d_v) * ke + d_v * kd
    txt = (f"rf {rf:.2f} % + β {beta:.2f} (unlevered {beta_u}, {beta_src}, D/E {d_e:.2f}) × ERP {erp} %"
           + (f" + Länderzuschlag {crp:.1f} pp" if crp else "")
           + f" = k_e {ke:.2f} %; k_d ({rf:.2f} + {spread:.2f}) × {1 - tax / 100:.2f} = {kd:.2f} %; D/V {d_v:.0%} ⇒ {w:.2f} %")
    return round(w, 2), txt


def shift(pats: list[str], dlt: float) -> list[dict]:
    """Ops: Muster (Glob) um *dlt* verschieben (alle Werte einer Verteilung)."""
    return [{"muster": p, "op": "+", "wert": dlt} for p in pats]


def set_(**kv) -> list[dict]:
    """Ops: Felder auf feste Werte setzen."""
    return [{"muster": k, "op": "=", "wert": v} for k, v in kv.items()]


def standard_szenarien(n_seg: int) -> list[tuple[str, list]]:
    tvg = [f"s{i}_tvg" for i in range(n_seg)]
    em = [f"s{i}_{p}" for i in range(n_seg) for p in ("em", "em_term")]
    return [
        ("WACC −1 pp (inkl. Holding)", shift(["s*_wacc", "bridge_cd"], -1.0)),
        ("WACC +1 pp (inkl. Holding)", shift(["s*_wacc", "bridge_cd"], 1.0)),
        ("TV-g −1 pp", shift(tvg, -1.0)),
        ("Margen −3 pp", shift(em, -3.0)),
    ]


def build_config(*, company, ccy, kurs, stichtag, stand, basis, bilanz, segments, bridge, corr, quellen, qv,
                 szenarien, sektor=None, kd=None, n_sim=50_000, seed=42) -> dict:
    """Konfiguration (Schema v2) aus Segmenten, Bridge und Korrelationen {(i, j): ρ}; fehlende Paare 0,5.
    *kd*: (Spread %, Steuersatz auf Zinsen %) wie in wacc(); das Portfolio rechnet damit den Hebel der Firma."""
    k = len(segments)
    c = {f"corr_{i}_{j}": 1.0 if i == j else corr.get((min(i, j), max(i, j)), 0.5) for i in range(k) for j in range(k)}
    return {
        "schema_version": 2, "saved_at": f"{stichtag}T00:00:00",
        "metadata": {"company": company, "report_currency": ccy, "stichtag": stichtag, "kurs_bei_recherche": kurs,
                     "basis_periode": basis, "bilanzstichtag": bilanz, "stand": stand,
                     **({"sektor": sektor} if sektor else {}),
                     **({"kd": {"spread_pct": kd[0], "steuer_pct": kd[1]}} if kd else {})},
        "ui_state": {
            "setup": {"setup_n_sim": n_sim, "setup_seed": seed, "setup_n_seg": k, "setup_mid_year": True,
                      "setup_ext_bridge": True, "setup_sampling": "Pseudo-Random (Standard)",
                      "setup_corr_enable": k > 1},
            "bridge": bridge, "segments": segments, "correlation": c},
        "quellen": quellen, "quellenverzeichnis": qv,
        "szenarien": [{"name": nm, "ops": o} for nm, o in szenarien],
    }


# ── Aktualisierung einer bestehenden Bewertung ──────────────────────────────────────────────────────────

def state_of(cfg: dict) -> dict:
    return {k: v for k, v in apply_config(cfg, {}).items() if k != "cfg_passthrough"}


def _write_back(old: dict, state: dict) -> dict:
    cfg = copy.deepcopy(old)
    ui = cfg["ui_state"]
    for block in [ui["setup"], ui["bridge"], ui.get("correlation", {}), *ui["segments"]]:
        for k in block:
            if k in state:
                block[k] = state[k]
    cfg.pop("ergebnis", None)
    return cfg


def update_config(old: dict, steps: list[tuple[str, list]], *, stichtag: str, kurs: float, stand: str,
                  metadata: dict | None = None, quellen: dict | None = None, qv: dict | None = None) -> dict:
    """Neue Version = alte + benannte Schritte (Ops). Die Schritte landen als ``ueberleitung`` (mit ``von``) im JSON."""
    st = state_of(old)
    for _, ops in steps:
        st, _ = apply_ops(st, ops)
    cfg = _write_back(old, st)
    cfg["saved_at"] = f"{stichtag}T00:00:00"
    cfg["metadata"].update(stichtag=stichtag, kurs_bei_recherche=kurs, stand=stand, **(metadata or {}))
    cfg["quellen"].update(quellen or {})
    cfg["quellenverzeichnis"].update(qv or {})
    cfg["ueberleitung"] = [{"name": n, "von": old["metadata"]["stichtag"], "ops": o} for n, o in steps]
    return cfg


def _fade(sg: dict, i: int, p: str = "rg") -> tuple[float, float, float]:
    """(λ, Startwert-Modus, Endwert-Modus) des Wachstums-Fades eines Segments."""
    m = lambda q: sg.get(f"s{i}_{q}_pert_mo", sg.get(f"s{i}_{q}_fixed", sg.get(f"s{i}_{q}_tri_mo",  # noqa: E731
                                                                                 sg.get(f"s{i}_{q}_n_mu"))))
    return sg[f"seg_{i}_fade_speed"], m(p), m("tvg")


def roll_quarter(old: dict, quarters: float = 1.0) -> tuple[str, list]:
    """Überleitungsschritt: Wachstumspfad um *quarters* Quartale verschieben, g0' = g_T + (g0 − g_T)·e^(−λ·q/4).
    Verschiebt alle Werte der Startverteilung um dasselbe Δ (Form bleibt). Nur Wachstum; Margen bleiben Urteil."""
    ops, txt = [], []
    for i, sg in enumerate(old["ui_state"]["segments"]):
        lam, g0, gt = _fade(sg, i)
        dlt = round((g0 - gt) * (math.exp(-lam * quarters / 4) - 1), 1)
        k = math.exp(-lam)
        txt.append(f"{gt + (g0 - gt) * k:.0f}→{gt + (g0 + dlt - gt) * k:.0f} %")
        ops += shift([f"s{i}_rg"], dlt)
    return f"Wachstumspfad um {quarters:g} Quartal(e) verschoben, Jahr 1: " + ", ".join(txt), ops


def plan_check(old: dict, base_old: list[float], base_new: list[float], quarters: float = 1.0) -> list[tuple]:
    """Ist-Basis gegen die Basis, die der alte Modalpfad nach *quarters* Quartalen erwartet.
    *base_old*: alte LTM je Segment (ggf. mit korrigierter Aufteilung). Rückgabe: (Name, alt, erwartet, Ist, Abw.)."""
    rows = []
    for i, (bo, bn) in enumerate(zip(base_old, base_new)):
        sg = old["ui_state"]["segments"][i]
        lam, g0, gt = _fade(sg, i)
        exp_ = bo * (1 + (gt + (g0 - gt) * math.exp(-lam)) / 100) ** (quarters / 4)
        rows.append((sg[f"seg_{i}_name"], bo, exp_, bn, bn / exp_ - 1))
    return rows


# ── Rechnen, prüfen, speichern ──────────────────────────────────────────────────────────────────────────

def evaluate(cfg: dict, kurs: float, *, old: dict | None = None, rdcf: list[tuple] | None = None,
             verbose: bool = True) -> dict:
    """Import-Check wie die App, Engine-Lauf, Checks, Szenarien, Reverse DCF und (mit *old*) Überleitung.
    *rdcf*: [(Name, Muster | None für alle WACC, lo, hi)]; Standard WACC −6…+20 pp. Schreibt ``ergebnis``."""
    out = print if verbose else (lambda *a, **k: None)
    parse_config_text(f"```json\n{json.dumps(cfg, ensure_ascii=False)}\n```")
    sc = build_simulation_config(cfg)
    r = MonteCarloEngine(sc).run()
    cfg["ergebnis"] = summarize_results(r, kurs=kurs)
    p, ccy = cfg["ergebnis"]["preis_je_aktie"], cfg["metadata"].get("report_currency", "")
    out(f"=== {cfg['metadata']['company']} · {cfg['metadata']['stand']} ===\n"
        f"P5/P50/P95 {p['p5']:.2f} / {p['p50']:.2f} / {p['p95']:.2f} {ccy} (Mittel {p['mean']:.2f}); Kurs {kurs}; "
        f"P(> Kurs) {np.mean(r.price_per_share > kurs):.1%}")
    for name, ev in r.segment_evs.items():
        out(f"  {name}: EV P50 {np.median(ev):,.0f}; TV/EV {np.median(r.segment_tv_ev_ratios[name]):.0%}")
    for ch in result_checks(sc, r, cfg["metadata"]):
        if ch.status != "ok" and "✅" not in str(ch.status):
            out(f"  {ch.status} C{ch.id} {ch.name}: {ch.detail}")
    if "irr" in cfg["ergebnis"]:
        i = cfg["ergebnis"]["irr"]
        out(f"  IRR P5/P50/P95 {i['p5']:.1%} / {i['p50']:.1%} / {i['p95']:.1%}")
    if old is not None:
        w = walk(old, cfg, named_steps_for(old, cfg))
        out(f"\n=== Überleitung {old['metadata']['stichtag']} → {cfg['metadata']['stichtag']} ===")
        for z in w["zeilen"]:
            out(f"  {z['schritt']:85} P50 {z['p50']:8.2f}" + ("" if z["delta"] is None else f"  Δ {z['delta']:+.2f}"))
        out(f"  schließt auf {w['p50_b']:.2f}: {w['schliesst']}")
        cfg["ergebnis"]["ueberleitung"] = w
    state = state_of(cfg)
    res = run_scenarios(state, cfg.get("szenarien", []))
    out("\n=== Szenarien ===")
    for z in res["zeilen"]:
        note = "; ".join(filter(None, [z["fehler"], "Δ mit Rauschen" if z["rauschen"] else "", *z["warnungen"]]))
        out(f"  {z['name']:80} P50 {z['p50']:8.2f} (Δ {z['delta']:+7.2f})" + (f"  [{note}]" if note else ""))
    rd_all = {}
    for name, pats, lo, hi in rdcf or [("WACC alle Segmente + Holding", None, -6.0, 20.0)]:
        pats = pats or preset_patterns("WACC alle Segmente + Holding", state)
        rd = reverse_dcf(state, pats, kurs, lo, hi)
        rd_all[name] = rd
        out(f"\n=== Reverse DCF ({name}) ===\n{rd.get('status')}: Δ {rd.get('delta', float('nan')):+.2f} pp "
            + ", ".join(f"{k} {v:.2f}" for k, v in (rd.get("implizit") or {}).items()) + (rd.get("detail") or ""))
    first = next(iter(rd_all))
    cfg["ergebnis"].update(szenarien=res, reverse_dcf={**rd_all[first], "hebel": first})
    return cfg


def save(cfg: dict, out_path: Path, vault_name: str | None = None) -> None:
    """JSON in den Analyse-Ordner und (mit *vault_name*) ins Bewertungsarchiv; kanonisch ist das Archiv."""
    text = json.dumps(cfg, ensure_ascii=False, indent=2)
    Path(out_path).write_text(text, encoding="utf-8")
    if vault_name:
        (VAULT / vault_name).write_text(text, encoding="utf-8")


def ui_check(cfg_path: str | Path) -> float:
    """UI-Abgleich: lädt die Konfiguration in die echten Streamlit-Seiten (AppTest), klickt „Simulation starten“
    und gibt den UI-P50 zurück (muss dem gespeicherten ``ergebnis`` auf den Cent entsprechen)."""
    from streamlit.testing.v1 import AppTest
    os.environ["SOTP_TEST_CFG"] = str(cfg_path)

    def _page():
        import json as _j
        import os as _o

        import streamlit as st

        from infrastructure.config_io import apply_config as _ac
        from presentation.pages.dcf_segments import render_segments
        from presentation.pages.dcf_setup import render_setup
        from presentation.pages.dcf_simulation import render_simulation
        if "_loaded" not in st.session_state:
            c = _j.load(open(_o.environ["SOTP_TEST_CFG"], encoding="utf-8"))
            for k, v in _ac(c, {}).items():
                st.session_state[k] = v
            st.session_state.update(_loaded=True, results=None, config=None)
        setup = render_setup(st.container())
        segs = render_segments(st.container(), int(setup["n_segments"]))
        render_simulation(st.container(), setup, segs)

    at = AppTest.from_function(_page, default_timeout=900)
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if "Simulation starten" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    return float(np.median(at.session_state["results"].price_per_share))
