"""Tests für domain.portfolio (Kelly/CRRA, Copula, Grenzen, Orders)."""
import numpy as np
import pytest

from domain.portfolio import (
    correlate_samples, horizon_returns, kelly_weights, market_world_returns, mix_worlds, nearest_corr,
    orders, wealth,
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


def test_frozen_position_keeps_weight_and_rest_is_reallocated():
    rng = np.random.default_rng(5)
    R = np.column_stack([rng.uniform(-0.9, -0.5, 5000), rng.uniform(0.3, 1.0, 5000), rng.uniform(0.2, 0.8, 5000)])
    lo = np.array([0.3, 0.0, 0.0]); hi = np.array([0.3, 0.5, 0.5])   # Titel 0 eingefroren trotz Verlust
    w = kelly_weights(R, fraction=1.0, min_cash=0.0, lo=lo, hi=hi)
    assert w[0] == 0.3 and w[1:].sum() == pytest.approx(0.7, abs=1e-4) and (w[1:] <= 0.5 + 1e-9).all()


def test_group_cap_limits_sum_of_group():
    rng = np.random.default_rng(8)
    R = rng.uniform(0.5, 1.5, (5000, 4))                        # alle attraktiv ⇒ Grenzen binden
    w = kelly_weights(R, fraction=1.0, max_w=0.3, group_caps=[([0, 1], 0.2)])
    assert w[0] + w[1] == pytest.approx(0.2, abs=1e-4) and w[2] == pytest.approx(0.3, abs=1e-4)


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


def test_market_world_mean_matches_rf_plus_erp():
    R = market_world_returns(200_000, np.array([0.25, 0.40]), np.eye(2), rf=0.03, erp=0.05, horizon=3, seed=1)
    assert R.mean(axis=0) == pytest.approx(np.full(2, 1.08 ** 3 - 1), abs=0.01)
    assert np.log1p(R[:, 1]).std() == pytest.approx(0.40 * np.sqrt(3), rel=0.02)


def test_mix_worlds_extremes_and_shared_groups():
    rng = np.random.default_rng(2)
    Rm, Rk = rng.normal(1.0, 0.1, (20_000, 3)), rng.normal(0.2, 0.1, (20_000, 3))
    assert np.array_equal(mix_worlds(Rm, Rk, 0.0, ["a", "b", "c"])[0], Rm)
    assert np.array_equal(mix_worlds(Rm, Rk, 1.0, ["a", "b", "c"])[0], Rk)
    R, mask = mix_worlds(Rm, Rk, 0.3, ["auto", "auto", "x"], seed=3)
    assert np.array_equal(mask[:, 0], mask[:, 1])                 # gleiche Gruppe ⇒ gemeinsam
    assert not np.array_equal(mask[:, 0], mask[:, 2])             # andere Gruppe ⇒ unabhängig
    assert mask.mean() == pytest.approx(0.3, abs=0.01)


def test_mixture_penalises_large_model_market_gap_under_kelly():
    # Zwei Titel mit gleicher Markt-Welt; A hat moderates, B riesiges Modell-Potenzial.
    rng = np.random.default_rng(4)
    n = 40_000
    Rmod = np.column_stack([rng.normal(0.35, 0.05, n), rng.normal(1.80, 0.10, n)])
    Rmkt = market_world_returns(n, np.array([0.25, 0.25]), np.eye(2), 0.03, 0.05, 3, seed=5)
    R, _ = mix_worlds(Rmod, Rmkt, 0.3, ["A", "B"], seed=6)
    w = kelly_weights(R, rf_h=1.03 ** 3 - 1, fraction=0.5, max_w=1.0)
    w_full_trust = kelly_weights(Rmod, rf_h=1.03 ** 3 - 1, fraction=0.5, max_w=1.0)
    assert w[1] < w_full_trust[1]                                   # B verliert Gewicht durch die Mischung


def test_wealth_cash_earns_rf():
    assert wealth(np.zeros((3, 2)), np.zeros(2), 0.08) == pytest.approx(np.full(3, 1.08))


def test_orders_never_overdraw_when_small_sell_is_dropped():
    # A (60 €) soll raus, Verkauf < Mindestorder 80 € entfällt; B soll 144 € kaufen, Cash nur 100 €
    # → Kauf auf 100 € (2 Stück) gekürzt, Konto bleibt ≥ 0
    shares, px = np.array([0.6, 0.0]), np.array([100.0, 50.0])
    delta, _ = orders(shares, px, 100.0, np.array([0.0, 0.9]), min_order_eur=80.0)
    assert delta.tolist() == [0.0, 2.0] and 100.0 - float(delta @ px) >= -1e-9


def test_orders_drop_smallest_buy_instead_of_scaling_all_below_min():
    # Cash 1.700 €, Käufe 1.200 € und 1.100 €, Mindestorder 1.000 €: anteilig wären beide < 1.000 €
    # → kleinerer Kauf entfällt, größerer läuft voll
    px = np.array([100.0, 100.0, 100.0])
    delta, _ = orders(np.array([0.0, 0.0, 33.0]), px, 1700.0, np.array([0.24, 0.22, 0.66]),
                      min_order_eur=1000.0, lot=1.0)
    assert delta.tolist() == [12.0, 0.0, 0.0]


def test_orders_fractional_by_default():
    # 2000 € × 20 % / 765,5 € = 0,522534… → 6 Stellen Richtung 0
    delta, target = orders(np.array([0.0]), np.array([765.5]), 2000.0, np.array([0.2]))
    assert target.tolist() == [0.522534] and target[0] * 765.5 <= 400.0


def test_orders_whole_shares_and_min_order():
    delta, target = orders(np.array([10.0, 0.0]), np.array([100.0, 30.0]), cash_eur=1000.0,
                           target_w=np.array([0.5, 0.49]), min_order_eur=50.0, lot=1.0)
    # Gesamt 2000 €: Titel 1 Ziel 10 Stück (keine Order), Titel 2 trunc(980/30) = 32
    assert delta.tolist() == [0.0, 32.0] and target.tolist() == [10.0, 32.0]
    delta, _ = orders(np.array([10.0]), np.array([100.0]), 0.0, np.array([0.98]), min_order_eur=150.0, lot=1.0)
    assert delta.tolist() == [0.0]                 # 1 Stück = 100 € < 150 €
    delta, target = orders(np.array([3.991219]), np.array([250.0]), 0.0, np.array([0.0]))
    assert delta.tolist() == [-3.991219] and target.tolist() == [0.0]   # Bruchstücke ganz verkaufen


def test_mix_worlds_rho_couples_groups_and_keeps_p():
    Rm, Rk = np.ones((40_000, 3)), np.zeros((40_000, 3))
    for rho, lo, hi in ((0.0, -0.02, 0.02), (0.4, 0.15, 0.35), (1.0, 0.999, 1.0)):
        _, mask = mix_worlds(Rm, Rk, 0.3, ["a", "b", "c"], seed=7, rho=rho)
        assert mask.mean() == pytest.approx(0.3, abs=0.01)               # p je Titel unverändert
        c = np.corrcoef(mask[:, 0], mask[:, 1])[0, 1]
        assert lo <= c <= hi, (rho, c)                                    # Kopplung wächst mit ρ
