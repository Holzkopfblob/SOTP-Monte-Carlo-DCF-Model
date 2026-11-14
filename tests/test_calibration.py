"""Tests für domain.calibration (PIT, Jahr-1-Wachstum, Aggregation, Zusammenfassung)."""
import numpy as np
import pytest

from domain.calibration import aggregate, pit, summarize, verdict, year1_growth


def test_pit_rank_and_ties():
    s = np.arange(100, dtype=float)
    assert pit(s, 49.5) == pytest.approx(0.50)
    assert pit(s, -1) == 0.0 and pit(s, 1000) == 1.0
    assert pit(np.ones(10), 1.0) == pytest.approx(0.5)


def test_pit_is_uniform_when_actuals_come_from_the_model():
    rng = np.random.default_rng(0)
    model = rng.normal(0.05, 0.02, 20_000)
    pits = [pit(model, a) for a in rng.normal(0.05, 0.02, 2_000)]
    s = summarize(pits)
    assert s["mittel"] == pytest.approx(0.5, abs=0.03) and s["im_90_band"] == pytest.approx(0.9, abs=0.03)
    assert verdict(s) == "keine systematische Abweichung erkennbar"


def test_verdict_detects_optimism_and_narrow_bands():
    rng = np.random.default_rng(1)
    model = rng.normal(0.10, 0.01, 20_000)                       # zu optimistisch und zu eng
    s = summarize([pit(model, a) for a in rng.normal(0.05, 0.03, 200)])
    assert "zu optimistisch" in verdict(s) and "Bänder zu eng" in verdict(s)


def test_year1_growth_fade_formula():
    g = year1_growth(np.array([0.30]), 0.02, 0.5, fade=True)
    assert g[0] == pytest.approx(0.02 + 0.28 * np.exp(-0.5))
    assert year1_growth(np.array([0.30]), 0.02, 0.5, fade=False)[0] == 0.30


def test_aggregate_revenue_weighted():
    g, m = aggregate([100.0, 300.0], [np.array([0.10]), np.array([0.0])], [np.array([0.20]), np.array([0.10])])
    assert g[0] == pytest.approx(10 / 400) and m[0] == pytest.approx((110 * 0.2 + 300 * 0.1) / 410)
