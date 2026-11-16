"""
Portfolio-Balancing auf Basis der SOTP-Bewertungsverteilungen.

Reine Funktionen ohne I/O. Ablauf:
  Bewertungs-Samples je Firma  →  correlate_samples (Iman-Conover)
  →  horizon_returns (R = max(V,0)/P − 1 über den Horizont)
  →  kelly_weights (CRRA mit γ = 1/Kelly-Faktor unter Grenzen)
  →  orders (ganze Stück, Mindestordergröße).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def nearest_corr(c: np.ndarray) -> np.ndarray:
    """Symmetrisch, Eigenwerte ≥ 0, Diagonale 1 (Eigenwert-Clip)."""
    c = (np.asarray(c, float) + np.asarray(c, float).T) / 2
    vals, vecs = np.linalg.eigh(c)
    c = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def correlate_samples(samples: np.ndarray, corr: np.ndarray, seed: int = 42) -> np.ndarray:
    """Iman-Conover: Spalten von *samples* (n×k) so umsortieren, dass ihre
    Rangkorrelation ≈ *corr* ist. Die Randverteilungen bleiben exakt erhalten."""
    n, k = samples.shape
    if k == 1:
        return samples.copy()
    z = np.random.default_rng(seed).standard_normal((n, k)) @ np.linalg.cholesky(nearest_corr(corr)).T
    ranks = z.argsort(axis=0).argsort(axis=0)
    return np.take_along_axis(np.sort(samples, axis=0), ranks, axis=0)


def horizon_returns(values: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Gesamtrendite bis zur Konvergenz Kurs → Wert. ``max(V, 0)``: beschränkte
    Haftung, eine Aktie fällt nicht unter 0 (LUMN/SPCX haben negative Samples)."""
    return np.maximum(values, 0.0) / np.asarray(prices, float) - 1.0


def market_world_returns(n: int, sigma: np.ndarray, corr: np.ndarray, rf: float, erp: float,
                         horizon: float, seed: int = 7) -> np.ndarray:
    """Welt „Markt hat recht“: Der Kurs ist fair, die Aktie verdient ihre Marktrendite rf + ERP p. a.
    Log-Rendite über den Horizont ~ N((ln(1+rf+erp) − σ²/2)·h, σ²·h), korreliert wie *corr*.
    *sigma* = annualisierte Volatilität je Titel (aus der Kurshistorie)."""
    sigma = np.asarray(sigma, float)
    k = len(sigma)
    z = np.random.default_rng(seed).standard_normal((n, k))
    if k > 1:
        z = z @ np.linalg.cholesky(nearest_corr(corr)).T
    mu = (np.log1p(rf + erp) - sigma ** 2 / 2) * horizon
    return np.expm1(mu + sigma * np.sqrt(horizon) * z)


def mix_worlds(R_model: np.ndarray, R_market: np.ndarray, p_market: float,
               groups: list[str], seed: int = 11, rho: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Szenario-Mischung: je Sample und Gruppe mit Wahrscheinlichkeit *p_market* die Markt-Welt,
    sonst die Modell-Welt. Titel derselben Gruppe wechseln gemeinsam (gemeinsamer Modellfehler,
    z. B. alle Autobauer). *rho* koppelt die Gruppen über einen gemeinsamen Faktor (Gauß-Copula):
    0 = unabhängig, 1 = ein Schalter für alle (Stilfehler des Modells, Querschnitt 16.11.2026).
    p_market bleibt je Gruppe exakt. Rückgabe: gemischte Renditen und Maske (True = Markt-Welt)."""
    from scipy.stats import norm
    n, k = R_model.shape
    labels = sorted(set(groups))
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, len(labels)))
    draws = z < norm.ppf(p_market) if 0 < p_market < 1 else np.full((n, len(labels)), p_market >= 1)
    col = [labels.index(g) for g in groups]
    mask = draws[:, col]
    return np.where(mask, R_market, R_model), mask


def wealth(R: np.ndarray, w: np.ndarray, rf_h: float) -> np.ndarray:
    """Endvermögen je Sample für Startvermögen 1; Cash (1 − Σw) verzinst mit rf_h."""
    return 1.0 + rf_h + (R - rf_h) @ w


def kelly_weights(
    R: np.ndarray,
    rf_h: float = 0.0,
    fraction: float = 0.5,
    max_w: float = 0.2,
    min_cash: float = 0.0,
    allow_short: bool = False,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    group_caps: list[tuple[list[int], float]] | None = None,
) -> np.ndarray:
    """Fractional Kelly als CRRA-Optimum mit γ = 1/fraction (γ = 1 ⇒ log ⇒ volles
    Kelly), direkt unter den Grenzen gelöst:
      0 ≤ w ≤ max_w  (mit Short: −max_w ≤ w ≤ max_w),  Σ|w| ≤ 1 − min_cash.
    *lo*/*hi* (je Titel) ersetzen diese Standardgrenzen, z. B. lo = hi = Ist-Gewicht
    für eine eingefrorene Position. *group_caps*: [(Titel-Indizes, Grenze)] ⇒ Σ|w| je Gruppe ≤ Grenze
    (eingefrorene Gewichte zählen mit; liegen sie schon darüber, gilt deren Summe als Grenze).
    Σ|w| über die Zerlegung w = u − v (u, v ≥ 0), damit alles glatt bleibt."""
    n, k = R.shape
    gamma = 1.0 / fraction
    X = R - rf_h
    budget = 1.0 - min_cash
    if budget <= 0:
        return np.zeros(k)

    def neg_utility(x):
        w = x[:k] - x[k:]
        W = np.maximum(1.0 + rf_h + X @ w, 1e-9)
        if abs(gamma - 1.0) < 1e-12:
            u, mu = np.log(W), 1.0 / W
        else:
            u, mu = W ** (1 - gamma) / (1 - gamma), W ** (-gamma)
        g = (X * mu[:, None]).mean(axis=0)
        return -u.mean(), -np.concatenate([g, -g])

    lo = np.full(k, -max_w if allow_short else 0.0) if lo is None else np.asarray(lo, float)
    hi = np.full(k, max_w) if hi is None else np.asarray(hi, float)
    b_lo = np.concatenate([np.maximum(lo, 0), np.maximum(-hi, 0)])     # u ≥ lo⁺, v ≥ (−hi)⁺
    b_hi = np.concatenate([np.maximum(hi, 0), np.maximum(-lo, 0)])     # u ≤ hi⁺, v ≤ (−lo)⁺
    bounds = list(zip(b_lo, b_hi))
    cons = [{"type": "ineq", "fun": lambda x: budget - x.sum(), "jac": lambda x: -np.ones(2 * k)}]
    for idx, cap in group_caps or []:
        m = np.zeros(2 * k)
        m[list(idx)] = m[[k + i for i in idx]] = 1.0
        cap = max(cap, float(m @ b_lo))                 # eingefrorene Mindestgewichte nicht verletzen
        cons.append({"type": "ineq", "fun": lambda x, m=m, c=cap: c - m @ x, "jac": lambda x, m=m: -m})
    res = minimize(neg_utility, b_lo.copy(), jac=True, method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    w = res.x[:k] - res.x[k:]
    fixed = np.isclose(lo, hi)
    w[fixed] = lo[fixed]                       # eingefrorene Titel exakt, ohne Solver-Rauschen
    w[~fixed & (np.abs(w) < 1e-6)] = 0.0
    return w


def orders(
    shares: np.ndarray,
    prices_eur: np.ndarray,
    cash_eur: float,
    target_w: np.ndarray,
    min_order_eur: float = 0.0,
    lot: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Order je Titel (Vorzeichen = Kauf/Verkauf) und Zielstück, in Vielfachen von *lot*
    (Standard: Bruchstücke auf 6 Stellen; lot=1 für ganze Aktien).
    Zielstück wird Richtung 0 gerundet, damit Käufe das Cash nicht überziehen.
    Orders unter *min_order_eur* entfallen (Zielstück = Bestand). Fehlt danach Cash für die
    Käufe (z. B. weil ein kleiner Verkauf entfallen ist), werden die Käufe anteilig gekürzt;
    fiele dabei einer unter die Mindestorder, entfällt der kleinste Kauf und der Rest wird neu
    verteilt. Der Vorschlag überzieht das Konto nie."""
    shares = np.asarray(shares, float)
    prices_eur = np.asarray(prices_eur, float)
    total = cash_eur + float(shares @ prices_eur)

    def to_lot(x):
        return np.round(np.trunc(x / lot) * lot, 6)

    def drop_small(d):
        d = d.copy()
        d[np.abs(d * prices_eur) < min_order_eur] = 0.0
        return d

    delta = drop_small(to_lot(target_w * total / prices_eur) - shares)
    avail = max(cash_eur - float(delta[delta < 0] @ prices_eur[delta < 0]), 0.0)  # Verkäufe bringen Cash
    # ponytail: „kleinster Kauf zuerst raus“ ist eine Heuristik; Optimum wäre eine Neuoptimierung mit Cash-Grenze
    while True:
        buys = delta > 0
        need = float(delta[buys] @ prices_eur[buys])
        if need <= avail + 1e-9:
            break
        scaled = to_lot(delta[buys] * avail / need)
        if np.all(scaled * prices_eur[buys] >= min_order_eur):
            delta[buys] = scaled
            break
        idx = np.flatnonzero(buys)
        delta[idx[np.argmin(delta[buys] * prices_eur[buys])]] = 0.0   # kleinsten Kauf streichen
    return delta, shares + delta
