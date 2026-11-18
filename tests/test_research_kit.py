"""Research-Kit: Quartal rollen (Zeitverschiebung des Fade-Pfads), Plan-Check, Aktualisierung mit Überleitung."""
import math

import pytest

from application.research_kit import (
    FEST, NORM, PERT, build_config, d, fade_start, plan_check, roll_quarter, segment, set_, state_of, update_config,
)
from application.scenario_service import apply_ops

LAM = 0.45


def _cfg(g1=70.0, tvg=3.0, rev=1000.0):
    g0 = fade_start(g1, tvg, LAM)
    seg = segment(0, "A", rev, 10, LAM, (PERT, g0 - 20, g0, g0 + 20), (FEST, 30.0), (FEST, 30.0), (FEST, 1.0),
                  (FEST, 1.0), (FEST, 20.0), (FEST, 20.0), (FEST, 2.0), (FEST, 2.0), (FEST, 10.0), (FEST, 10.0),
                  (NORM, 9.0, 0.5), (PERT, 2.5, tvg, 3.5))
    bridge = {**d("bridge_cc", FEST, 0.0), **d("bridge_cd", FEST, 9.0), **d("bridge_nd", FEST, 0.0),
              **d("bridge_sh", FEST, 100.0), "setup_ext_bridge": True}
    return build_config(company="Test AG", ccy="EUR", kurs=10.0, stichtag="2026-11-01", stand="t", basis="LTM",
                        bilanz="2026-09-30", segments=[seg], bridge=bridge, corr={}, quellen={}, qv={}, szenarien=[], kd=(0.0, 25.0))


def _g(cfg, t):
    s = cfg["ui_state"]["segments"][0]
    g0, gt = s["s0_rg_pert_mo"], s["s0_tvg_pert_mo"]
    return gt + (g0 - gt) * math.exp(-LAM * t)


def test_roll_quarter_ist_zeitverschiebung():
    old = _cfg()
    name, ops = roll_quarter(old)
    st, _ = apply_ops(state_of(old), ops)
    new_g0 = st["s0_rg_pert_mo"]
    gt = old["ui_state"]["segments"][0]["s0_tvg_pert_mo"]
    for t in (1, 2, 5):                                   # g'(t) = g(t + ¼), bis auf Rundung des Δ auf 0,1
        assert gt + (new_g0 - gt) * math.exp(-LAM * t) == pytest.approx(_g(old, t + 0.25), abs=0.05)
    assert st["s0_rg_pert_hi"] - st["s0_rg_pert_lo"] == pytest.approx(40.0)   # Form der Verteilung bleibt
    assert "70→63 %" in name


def test_plan_check_null_bei_basis_auf_plan():
    old = _cfg(g1=70.0)
    exp_ = 1000.0 * 1.70 ** 0.25
    (_, bo, ex, bn, dv), = plan_check(old, [1000.0], [exp_])
    assert ex == pytest.approx(exp_) and dv == pytest.approx(0.0)


def test_update_config_schreibt_ueberleitung_und_metadaten():
    old = _cfg()
    cfg = update_config(old, [("b) Basis", set_(seg_0_basrev=1200.0))], stichtag="2026-11-18", kurs=11.0,
                        stand="nach Q3", quellen={"seg_0_basrev": "neu"})
    assert cfg["ui_state"]["segments"][0]["seg_0_basrev"] == 1200.0
    assert cfg["ueberleitung"][0]["von"] == "2026-11-01" and cfg["metadata"]["kurs_bei_recherche"] == 11.0
    assert old["ui_state"]["segments"][0]["seg_0_basrev"] == 1000.0           # alte Version unverändert
