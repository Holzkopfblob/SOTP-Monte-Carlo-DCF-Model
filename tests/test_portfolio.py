"""Tests für domain.portfolio (Kelly/CRRA, Copula, Grenzen, Orders)."""
import numpy as np
import pytest

from domain.portfolio import (
    correlate_samples, horizon_returns, kelly_weights, nearest_corr, orders, wealth,
)


def _binary(p, gain, loss, n=10_000):
    k = int(round(p * n))
    return np.concatenate([np.full(k, gain), np.full(n - k, -loss)])[:, None]


def test_full_kelly_matches_binary_formula():
    # f* = p/a − q/b, a = Verlustanteil, b = Gewinnanteil
    R = _binary(0.6, gain=1.0, loss=0.5)
    w = kelly_weights(R, fraction=1.0, max_w=1.0)
    assert w[0] == pytest.approx(0.6 / 0.5 - 0.4 / 1.0, abs=1e-3)   # = 0.8


def test_full_kelly_interior_solution():
    R = _binary(0.6, gain=1.0, loss=1.0)          # f* = 0.6 − 0.4 = 0.2
    assert kelly_weights(R, fraction=1.0, max_w=1.0)[0] == pytest.approx(0.2, abs=1e-3)


def test_half_kelly_is_smaller_than_full():
    R = _binary(0.6, gain=1.0, loss=1.0)
    half = kelly_weights(R, fraction=0.5, max_w=1.0)[0]
    assert 0.05 < half < 0.2


def test_no_loss_sample_stays_finite_at_cap():
    R = np.random.default_rng(0).uniform(0.1, 1.0, (5000, 1))
    assert kelly_weights(R, fraction=0.5, max_w=0.2)[0] == pytest.approx(0.2, abs=1e-6)


def test_negative_value_sample_gives_total_loss_not_nan():
    R = horizon_returns(np.array([[-4.0], [10.0]]), np.array([5.0]))
    assert R[0, 0] == -1.0 and np.isfinite(R).all()


def test_limits_hold_long_only():
    rng = np.random.default_rng(1)
    R = rng.uniform(0.2, 1.5, (5000, 5))           # alles attraktiv → Budget bindet
    w = kelly_weights(R, fraction=1.0, max_w=0.3, min_cash=0.1)
    assert (w >= 0).all() and (w <= 0.3 + 1e-9).all()
    assert w.sum() == pytest.approx(0.9, abs=1e-4)


def test_negative_upside_zero_without_short_negative_with_short():
    R = np.random.default_rng(2).uniform(-0.95, -0.5, (5000, 1))
    assert kelly_weights(R, fraction=0.5, max_w=0.2)[0] == 0.0
    w = kelly_weights(R, fraction=0.5, max_w=0.2, allow_short=True)[0]
    assert w == pytest.approx(-0.2, abs=1e-6)


def test_short_respects_gross_budget():
    rng = np.random.default_rng(3)
    R = np.column_stack([rng.uniform(0.5, 1.0, 5000), rng.uniform(-0.9, -0.6, 5000)])
    w = kelly_weights(R, fraction=1.0, max_w=0.6, min_cash=0.2, allow_short=True)
    assert np.abs(w).sum() <= 0.8 + 1e-6 and w[0] > 0 > w[1]


def test_copula_keeps_marginals_and_hits_correlation():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.lognormal(0, 0.5, 20_000), rng.normal(5, 1, 20_000)])
    target = np.array([[1.0, 0.6], [0.6, 1.0]])
    Y = correlate_samples(X, target, seed=7)
    assert np.array_equal(np.sort(Y, axis=0), np.sort(X, axis=0))
    ranks = Y.argsort(0).argsort(0)
    assert np.corrcoef(ranks.T)[0, 1] == pytest.approx(0.6, abs=0.02)


def test_nearest_corr_repairs_non_psd():
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    c = nearest_corr(bad)
    assert np.linalg.eigvalsh(c).min() > -1e-10 and np.allclose(np.diag(c), 1)


def test_wealth_cash_earns_rf():
    assert wealth(np.zeros((3, 2)), np.zeros(2), 0.08) == pytest.approx(np.full(3, 1.08))


def test_orders_whole_shares_and_min_order():
    delta, target = orders(np.array([10.0, 0.0]), np.array([100.0, 30.0]), cash_eur=1000.0,
                           target_w=np.array([0.5, 0.49]), min_order_eur=50.0)
    # Gesamt 2000 €: Titel 1 Ziel 10 Stück (keine Order), Titel 2 trunc(980/30) = 32
    assert delta.tolist() == [0.0, 32.0] and target.tolist() == [10.0, 32.0]
    delta, _ = orders(np.array([10.0]), np.array([100.0]), 0.0, np.array([0.98]), min_order_eur=150.0)
    assert delta.tolist() == [0.0]                 # 1 Stück = 100 € < 150 €
    delta, target = orders(np.array([3.991219]), np.array([250.0]), 0.0, np.array([0.0]))
    assert delta.tolist() == [-3.991219] and target.tolist() == [0.0]   # Bruchstücke ganz verkaufen
