"""Robustes Kelly (Hansen–Sargent-Entropie-Strafe): kleine, aber positive Gewichte für schwach unterbewertete Titel."""
import numpy as np
import pytest

from domain.portfolio import kelly_weights, robust_anchor


def _R(mus, n=20000, sig=0.25, h=10, seed=1, rho=0.8):
    rng = np.random.default_rng(seed)        # gemeinsamer Faktor: hoch korrelierte Titel ⇒ Kelly-Ecklösung
    z = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, len(mus)))
    return np.expm1((np.log1p(np.array(mus)) - sig ** 2 / 2) * h + sig * np.sqrt(h) * z)


def test_anker_summe_eins_und_nur_positiver_vorsprung():
    a = robust_anchor(np.array([0.04, 0.01, -0.02, np.nan]), budget=1.0)
    assert a.sum() == pytest.approx(1.0)
    assert a[2] == a[3] == 0.0 and a[0] == pytest.approx(4 * a[1])
    assert a[-1] == pytest.approx(0.02)


def test_kappa_null_ist_klassisches_kelly():
    R = _R([0.14, 0.09, 0.085])
    assert np.allclose(kelly_weights(R, 0.03 * 10, 0.5, 1.0, kappa=0.0),
                       kelly_weights(R, 0.03 * 10, 0.5, 1.0), atol=1e-12)


def test_robust_gibt_jedem_unterbewerteten_etwas_und_monoton():
    mus = [0.16, 0.11, 0.095]                        # klar, mittel, knapp über der Hürde 0,09
    R, rf_h = _R(mus), 1.03 ** 10 - 1
    edge = np.array(mus) - 0.09
    plain = kelly_weights(R, rf_h, 0.5, 1.0)
    assert plain[2] < 1e-6                           # klassisch: knapper Titel fällt raus (Ecklösung)
    w = kelly_weights(R, rf_h, 0.5, 1.0, kappa=0.01, anchor=robust_anchor(edge, 1.0), horizon=10)
    assert np.all(w > 1e-3) and w[0] > w[1] > w[2]
    assert w.sum() <= 1.0 + 1e-9


def test_eingefroren_und_hi_null_bleiben_respektiert():
    R, rf_h = _R([0.16, 0.11, 0.05]), 1.03 ** 10 - 1
    lo, hi = np.array([0.0, 0.1, 0.0]), np.array([1.0, 0.1, 0.0])   # Titel 2 überbewertet ⇒ hi 0
    w = kelly_weights(R, rf_h, 0.5, 1.0, lo=lo, hi=hi, kappa=0.02,
                      anchor=robust_anchor(np.array([0.07, 0.02, -0.04]), 1.0), horizon=10)
    assert w[1] == pytest.approx(0.1) and w[2] == pytest.approx(0.0, abs=1e-9)


def test_robust_ohne_leerverkauf():
    with pytest.raises(ValueError):
        kelly_weights(_R([0.1]), kappa=0.01, allow_short=True)
