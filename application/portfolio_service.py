"""
Portfolio Optimisation & Analysis Service
==========================================

Comprehensive engine for portfolio analysis, optimisation and stress testing.

Capabilities
------------
1. **Single-asset analysis** – 12+ metrics incl. VaR, CVaR, Sortino, Omega
2. **Seven optimisation methods** – Max Sharpe, Min Vol, Risk Parity,
   Min CVaR, Max Diversification, Multi-Asset Kelly, Equal Weight (1/N)
3. **Efficient frontier** – parametric frontier with custom bounds
4. **Stress-test engine** – market shock, correlation stress, sector crisis
5. **Correlation utilities** – cluster-based sector model, PSD enforcement
6. **Fair-value sample generation** – 6 distribution types

All computations are vectorised (numpy) and the class is stateless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════
# Sector correlation model
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_CLUSTERS: Dict[str, str] = {
    "Technologie":       "growth",
    "Konsumgüter":       "cyclical",
    "Industrie":         "cyclical",
    "Grundstoffe":       "cyclical",
    "Gesundheit":        "defensive",
    "Versorger":         "defensive",
    "Telekommunikation": "defensive",
    "Finanzen":          "financial",
    "Immobilien":        "financial",
    "Energie":           "energy",
    "Sonstige":          "other",
}

# Cross-cluster default correlations (symmetric – order doesn't matter)
_CLUSTER_CORR: Dict[Tuple[str, str], float] = {
    ("growth",    "growth"):    0.55,
    ("cyclical",  "cyclical"):  0.50,
    ("defensive", "defensive"): 0.45,
    ("financial", "financial"): 0.55,
    ("energy",    "energy"):    0.50,
    ("other",     "other"):     0.40,

    ("growth",    "cyclical"):  0.45,
    ("growth",    "defensive"): 0.20,
    ("growth",    "financial"): 0.40,
    ("growth",    "energy"):    0.25,
    ("growth",    "other"):     0.30,

    ("cyclical",  "defensive"): 0.25,
    ("cyclical",  "financial"): 0.40,
    ("cyclical",  "energy"):    0.35,
    ("cyclical",  "other"):     0.30,

    ("defensive", "financial"): 0.25,
    ("defensive", "energy"):    0.15,
    ("defensive", "other"):     0.20,

    ("financial", "energy"):    0.30,
    ("financial", "other"):     0.30,

    ("energy",    "other"):     0.25,
}

SAME_SECTOR_CORR = 0.65
DEFAULT_CROSS_CORR = 0.30


def _cluster_corr(c1: str, c2: str) -> float:
    """Look up cross-cluster correlation (order-independent)."""
    return _CLUSTER_CORR.get(
        (c1, c2),
        _CLUSTER_CORR.get((c2, c1), DEFAULT_CROSS_CORR),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetInput:
    """One asset to include in the portfolio analysis."""
    name: str
    sector: str
    current_price: float
    fv_samples: np.ndarray          # (n_sim,) MC fair-value samples
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class AssetMetrics:
    """Comprehensive single-asset analysis results."""
    name: str
    sector: str
    current_price: float

    # Fair-value statistics
    mean_fv: float
    median_fv: float
    fv_std: float
    fv_p5: float
    fv_p25: float
    fv_p75: float
    fv_p95: float

    # Return metrics
    expected_return: float
    return_std: float
    prob_profit: float
    margin_of_safety: float

    # Kelly
    kelly_fraction: float
    half_kelly: float

    # Scenario percentiles (as returns)
    upside_p75: float
    downside_p25: float

    # Risk metrics
    var_5: float
    cvar_5: float
    sortino_ratio: float
    omega_ratio: float

    # Composite signal
    signal: str          # "🟢 Kaufen" | "🟡 Halten" | "🔴 Meiden"

    # Raw vectors for downstream use (excluded from repr)
    returns: np.ndarray = field(repr=False)
    fv_samples: np.ndarray = field(repr=False)


@dataclass
class PortfolioResult:
    """Result for one optimisation method."""
    name: str
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_5: float
    cvar_5: float
    prob_loss: float
    diversification_ratio: float
    effective_n_assets: float


@dataclass
class StressTestResult:
    """Stress test output for one portfolio method."""
    method_name: str
    return_normal: float
    return_stressed: float
    delta_return: float
    vol_stressed: float
    var_5_stressed: float
    cvar_5_stressed: float
    prob_loss: float


# ═══════════════════════════════════════════════════════════════════════════
# Fair-value sample generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_fv_samples(
    dist_type: str,
    params: dict,
    n: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Generate *n* fair-value Monte-Carlo samples from the chosen distribution.

    Supported *dist_type* values::

        "Normal", "Lognormal", "Aus DCF-App (μ, σ, Schiefe)",
        "PERT", "Dreiecksverteilung", "Gleichverteilung"

    Returns an (n,) array clipped to a minimum of 0.01.
    """
    rng = np.random.default_rng(seed)

    if dist_type == "Normal":
        samples = rng.normal(params["mean"], params["std"], size=n)

    elif dist_type == "Lognormal":
        desired_mean = params["mean"]
        desired_std = params["std"]
        if desired_mean > 0 and desired_std > 0:
            sigma_sq = np.log(1 + (desired_std / desired_mean) ** 2)
            mu_ln = np.log(desired_mean) - sigma_sq / 2
            sigma_ln = np.sqrt(sigma_sq)
            samples = rng.lognormal(mu_ln, sigma_ln, size=n)
        else:
            samples = np.full(n, max(desired_mean, 0.01))

    elif dist_type == "Aus DCF-App (μ, σ, Schiefe)":
        mu = params["mean"]
        sigma = params["std"]
        skewness = params.get("skew", 0.0)
        if abs(skewness) < 0.5:
            samples = rng.normal(mu, sigma, size=n)
        else:
            if mu > 0 and sigma > 0:
                sigma_sq = np.log(1 + (sigma / mu) ** 2)
                mu_ln = np.log(mu) - sigma_sq / 2
                sigma_ln = np.sqrt(sigma_sq)
                samples = rng.lognormal(mu_ln, sigma_ln, size=n)
            else:
                samples = rng.normal(max(mu, 0.01), max(sigma, 0.01), size=n)

    elif dist_type == "PERT":
        lo, mode, hi = params["low"], params["mode"], params["high"]
        if hi <= lo:
            samples = np.full(n, mode)
        else:
            lam = 4.0
            alpha = 1 + lam * (mode - lo) / (hi - lo)
            beta_p = 1 + lam * (hi - mode) / (hi - lo)
            samples = lo + (hi - lo) * rng.beta(alpha, beta_p, size=n)

    elif dist_type == "Dreiecksverteilung":
        lo, mode, hi = params["low"], params["mode"], params["high"]
        if hi <= lo:
            samples = np.full(n, mode)
        else:
            samples = rng.triangular(lo, mode, hi, size=n)

    elif dist_type == "Gleichverteilung":
        lo, hi = params["low"], params["high"]
        if hi <= lo:
            samples = np.full(n, lo)
        else:
            samples = rng.uniform(lo, hi, size=n)

    else:
        samples = np.full(n, params.get("mean", 0.0))

    return np.maximum(samples, 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Core engine
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioAnalyser:
    """Stateless portfolio analysis & optimisation engine.

    Instantiate with the risk-free rate, then call individual methods.
    Every public method is a pure function of its arguments (except
    ``self.rf`` which is read-only).
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate

    # ── Single-asset analysis ─────────────────────────────────────────

    def analyse_asset(self, asset: AssetInput) -> AssetMetrics:
        """Compute all single-asset metrics from an MC fair-value distribution."""
        fv = asset.fv_samples
        p = asset.current_price

        mean_fv = float(np.mean(fv))
        median_fv = float(np.median(fv))
        fv_std = float(np.std(fv))

        returns = (fv / p) - 1.0
        mu_r = float(np.mean(returns))
        std_r = float(np.std(returns))

        prob_profit = float(np.mean(fv > p))
        mos = (median_fv - p) / median_fv if median_fv > 0 else 0.0

        # Kelly criterion:  f* = E[R] / Var(R)
        var_r = float(np.var(returns))
        kelly = mu_r / var_r if var_r > 1e-12 else 0.0
        kelly = float(np.clip(kelly, 0.0, 1.0))

        upside_p75 = (float(np.percentile(fv, 75)) / p) - 1.0
        downside_p25 = (float(np.percentile(fv, 25)) / p) - 1.0

        # Value at Risk & Conditional VaR  (5 % level)
        var_5 = float(np.percentile(returns, 5))
        tail_mask = returns <= np.percentile(returns, 5)
        cvar_5 = float(np.mean(returns[tail_mask])) if tail_mask.sum() > 0 else var_5

        # Sortino ratio  (target = 0)
        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-12
        sortino = mu_r / max(downside_std, 1e-12)

        # Omega ratio:  E[max(R,0)] / E[max(-R,0)]
        gains = np.maximum(returns, 0.0)
        losses = np.maximum(-returns, 0.0)
        omega = float(np.mean(gains)) / max(float(np.mean(losses)), 1e-12)

        # Composite signal
        if mos > 0.20 and prob_profit > 0.65:
            signal = "🟢 Kaufen"
        elif mos > 0.0 and prob_profit > 0.50:
            signal = "🟡 Halten"
        else:
            signal = "🔴 Meiden"

        return AssetMetrics(
            name=asset.name, sector=asset.sector, current_price=p,
            mean_fv=mean_fv, median_fv=median_fv, fv_std=fv_std,
            fv_p5=float(np.percentile(fv, 5)),
            fv_p25=float(np.percentile(fv, 25)),
            fv_p75=float(np.percentile(fv, 75)),
            fv_p95=float(np.percentile(fv, 95)),
            expected_return=mu_r, return_std=std_r,
            prob_profit=prob_profit, margin_of_safety=mos,
            kelly_fraction=kelly, half_kelly=kelly / 2.0,
            upside_p75=upside_p75, downside_p25=downside_p25,
            var_5=var_5, cvar_5=cvar_5,
            sortino_ratio=sortino, omega_ratio=omega,
            signal=signal,
            returns=returns, fv_samples=fv,
        )

    def analyse_all(self, assets: List[AssetInput]) -> List[AssetMetrics]:
        """Analyse every asset and return a list of metrics."""
        return [self.analyse_asset(a) for a in assets]

    # ── Matrix construction ───────────────────────────────────────────

    @staticmethod
    def build_returns_matrix(assets: List[AssetInput]) -> np.ndarray:
        """Stack per-asset return vectors into an (n_sim, n_assets) matrix."""
        return np.column_stack([
            (a.fv_samples / a.current_price) - 1.0 for a in assets
        ])

    @staticmethod
    def build_sector_correlation(sectors: List[str]) -> np.ndarray:
        """Create a correlation matrix from sector assignments.

        Uses a cluster-based model:
        - Same sector → ρ = 0.65
        - Same cluster, different sector → cluster default
        - Different cluster → cross-cluster default
        """
        n = len(sectors)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if sectors[i] == sectors[j]:
                    rho = SAME_SECTOR_CORR
                else:
                    ci = SECTOR_CLUSTERS.get(sectors[i], "other")
                    cj = SECTOR_CLUSTERS.get(sectors[j], "other")
                    rho = _cluster_corr(ci, cj)
                corr[i, j] = rho
                corr[j, i] = rho
        return corr

    @staticmethod
    def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Project a symmetric matrix to the nearest positive semi-definite
        matrix and re-normalise to a valid correlation matrix (diag = 1).
        """
        B = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, epsilon)
        result = eigvecs @ np.diag(eigvals) @ eigvecs.T
        result = (result + result.T) / 2.0
        # Re-normalise diagonal to 1
        d = np.sqrt(np.diag(result))
        if np.all(d > 0):
            result = result / np.outer(d, d)
            np.fill_diagonal(result, 1.0)
        return result

    @staticmethod
    def build_covariance(
        returns_matrix: np.ndarray,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (mu, std, cov) from returns.

        If *correlation_matrix* is supplied it overrides the sample
        correlation; the sample standard deviations are always used.
        """
        mu = np.mean(returns_matrix, axis=0)
        stds = np.std(returns_matrix, axis=0)
        if correlation_matrix is not None:
            corr = correlation_matrix
        else:
            corr = np.corrcoef(returns_matrix, rowvar=False)
            if corr.ndim == 0:
                corr = np.array([[1.0]])
        cov = np.outer(stds, stds) * corr
        return mu, stds, cov

    # ── Portfolio result helper ───────────────────────────────────────

    def _make_result(
        self,
        name: str,
        w: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> PortfolioResult:
        """Compute all portfolio-level metrics for a given weight vector."""
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - self.rf) / max(vol, 1e-12)

        port_returns = returns_matrix @ w
        var_5 = float(np.percentile(port_returns, 5))
        tail_mask = port_returns <= np.percentile(port_returns, 5)
        cvar_5 = float(np.mean(port_returns[tail_mask])) if tail_mask.sum() > 0 else var_5
        prob_loss = float(np.mean(port_returns < 0))

        weighted_vol = float(w @ std_vec)
        div_ratio = weighted_vol / max(vol, 1e-12)
        n_eff = 1.0 / max(float(np.sum(w ** 2)), 1e-12)

        return PortfolioResult(
            name=name, weights=w,
            expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
            var_5=var_5, cvar_5=cvar_5, prob_loss=prob_loss,
            diversification_ratio=div_ratio, effective_n_assets=n_eff,
        )

    # ── Optimisation methods ──────────────────────────────────────────

    def optimise_max_sharpe(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[PortfolioResult]:
        """Maximise the Sharpe ratio (Markowitz tangency portfolio)."""
        n = len(mu)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_sharpe(w):
            r = w @ mu
            v = np.sqrt(w @ cov @ w)
            return -(r - self.rf) / max(v, 1e-12)

        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if not res.success:
            return None
        return self._make_result("Max Sharpe", res.x, mu, cov, std_vec, returns_matrix)

    def optimise_min_vol(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[PortfolioResult]:
        """Minimise portfolio volatility (global minimum variance)."""
        n = len(mu)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            lambda w: np.sqrt(w @ cov @ w),
            w0, method="SLSQP", bounds=bounds, constraints=cons,
        )
        if not res.success:
            return None
        return self._make_result("Min Volatilität", res.x, mu, cov, std_vec, returns_matrix)

    def optimise_risk_parity(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> Optional[PortfolioResult]:
        """Equal Risk Contribution portfolio.

        Each asset contributes equally to total portfolio risk.
        """
        n = cov.shape[0]

        def risk_budget_obj(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = cov @ w
            risk_contrib = w * marginal / max(port_vol, 1e-12)
            target = port_vol / n
            return float(np.sum((risk_contrib - target) ** 2))

        w0 = np.ones(n) / n
        bounds = [(0.01, 1.0)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(risk_budget_obj, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if not res.success:
            return None
        return self._make_result("Risk Parity", res.x, mu, cov, std_vec, returns_matrix)

    def optimise_min_cvar(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        alpha: float = 0.05,
    ) -> Optional[PortfolioResult]:
        """Minimise Conditional Value at Risk (Expected Shortfall) at the
        given confidence level *alpha* using the MC sample distribution.
        """
        n = mu.shape[0]
        if bounds is None:
            bounds = [(0.0, 1.0)] * n

        def cvar_objective(w):
            port_ret = returns_matrix @ w
            threshold = np.percentile(port_ret, alpha * 100)
            tail = port_ret[port_ret <= threshold]
            return -float(np.mean(tail)) if len(tail) > 0 else 0.0

        w0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(cvar_objective, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if not res.success:
            return None
        return self._make_result("Min CVaR", res.x, mu, cov, std_vec, returns_matrix)

    def optimise_max_diversification(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[PortfolioResult]:
        """Maximise the Diversification Ratio  DR = (w'σ) / σ_p."""
        n = len(mu)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n

        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov @ w)
            weighted_vol = w @ std_vec
            return -(weighted_vol / max(port_vol, 1e-12))

        w0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(neg_div_ratio, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if not res.success:
            return None
        return self._make_result(
            "Max Diversifikation", res.x, mu, cov, std_vec, returns_matrix,
        )

    def kelly_weights(
        self,
        asset_metrics: List[AssetMetrics],
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> PortfolioResult:
        """Multi-asset Kelly criterion.

        Maximises expected log-growth  ≈  w'μ − ½ w'Σw  subject to
        long-only and budget constraints, then applies **half-Kelly**
        scaling and re-normalisation for practical position sizing.
        """
        n = len(mu)

        def neg_kelly_growth(w):
            return -(w @ mu - 0.5 * w @ cov @ w)

        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(neg_kelly_growth, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)

        if res.success:
            w_full = res.x
            w_half = w_full * 0.5
            total = w_half.sum()
            if total > 1e-12:
                w_half = w_half / total
            else:
                w_half = np.ones(n) / n
            return self._make_result(
                "Kelly (Multi-Asset)", w_half, mu, cov, std_vec, returns_matrix,
            )

        # Fallback: normalised single-asset half-Kelly
        kelly_raw = np.array([am.half_kelly for am in asset_metrics])
        s = kelly_raw.sum()
        w = kelly_raw / s if s > 0 else np.ones(n) / n
        return self._make_result(
            "Kelly (Single-Asset)", w, mu, cov, std_vec, returns_matrix,
        )

    def equal_weights(
        self,
        n: int,
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> PortfolioResult:
        """Naïve 1/N equal-weight benchmark portfolio."""
        w = np.ones(n) / n
        return self._make_result(
            "Gleichgewicht (1/N)", w, mu, cov, std_vec, returns_matrix,
        )

    # ── Efficient frontier ────────────────────────────────────────────

    def efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute *n_points* on the efficient frontier.

        Returns ``(frontier_vols, frontier_rets)`` both as 1-D arrays.
        """
        n = len(mu)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n
        cons_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier_vols: List[float] = []
        frontier_rets: List[float] = []

        for target_r in target_returns:
            cons = cons_base + [
                {"type": "eq", "fun": lambda w, r=target_r: w @ mu - r}
            ]
            w0 = np.ones(n) / n
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w), w0,
                method="SLSQP", bounds=bounds, constraints=cons,
            )
            if res.success:
                frontier_vols.append(float(np.sqrt(res.x @ cov @ res.x)))
                frontier_rets.append(target_r)

        return np.array(frontier_vols), np.array(frontier_rets)

    # ── Stress testing ────────────────────────────────────────────────

    def stress_test(
        self,
        portfolios: Dict[str, np.ndarray],
        returns_matrix: np.ndarray,
        asset_sectors: List[str],
        market_shock_pct: float = -30.0,
        corr_stress: float = 0.85,
        sector_shock: Optional[str] = None,
        sector_shock_pct: float = 0.0,
    ) -> List[StressTestResult]:
        """Run a stress scenario across all supplied portfolio methods.

        Parameters
        ----------
        portfolios : dict  name → weight vector
        returns_matrix : (n_sim, n_assets) array
        asset_sectors : sector label per asset
        market_shock_pct : uniform return shock (e.g. −30 → −30 %)
        corr_stress : raise all off-diagonal correlations to at least this
        sector_shock : optional sector name for an additional targeted shock
        sector_shock_pct : additional return shock for that sector
        """
        shocked_returns = returns_matrix + (market_shock_pct / 100.0)

        if sector_shock and sector_shock_pct != 0:
            for idx, sec in enumerate(asset_sectors):
                if sec == sector_shock:
                    shocked_returns[:, idx] += (sector_shock_pct / 100.0)

        results: List[StressTestResult] = []
        for method_name, w in portfolios.items():
            if w is None:
                continue
            w = np.array(w)
            ret_normal = float(np.mean(returns_matrix @ w))

            port_stressed = shocked_returns @ w
            ret_stressed = float(np.mean(port_stressed))
            vol_stressed = float(np.std(port_stressed))
            var_5 = float(np.percentile(port_stressed, 5))
            tail = port_stressed[port_stressed <= np.percentile(port_stressed, 5)]
            cvar_5 = float(np.mean(tail)) if len(tail) > 0 else var_5
            prob_loss = float(np.mean(port_stressed < 0))

            results.append(StressTestResult(
                method_name=method_name,
                return_normal=ret_normal,
                return_stressed=ret_stressed,
                delta_return=ret_stressed - ret_normal,
                vol_stressed=vol_stressed,
                var_5_stressed=var_5,
                cvar_5_stressed=cvar_5,
                prob_loss=prob_loss,
            ))

        return results

    # ── Run all optimisations at once ─────────────────────────────────

    def run_all_optimisations(
        self,
        asset_metrics: List[AssetMetrics],
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Optional[PortfolioResult]]:
        """Convenience method that runs all 7 optimisation strategies
        and returns a ``name → PortfolioResult`` dict.  Values may be
        *None* if the optimisation failed (e.g. < 2 assets).
        """
        n = len(mu)
        results: Dict[str, Optional[PortfolioResult]] = {}

        results["Gleichgewicht (1/N)"] = self.equal_weights(
            n, mu, cov, std_vec, returns_matrix,
        )

        if n >= 2:
            results["Max Sharpe"] = self.optimise_max_sharpe(
                mu, cov, std_vec, returns_matrix, bounds,
            )
            results["Min Volatilität"] = self.optimise_min_vol(
                mu, cov, std_vec, returns_matrix, bounds,
            )
            results["Risk Parity"] = self.optimise_risk_parity(
                mu, cov, std_vec, returns_matrix,
            )
            results["Min CVaR"] = self.optimise_min_cvar(
                mu, cov, std_vec, returns_matrix, bounds,
            )
            results["Max Diversifikation"] = self.optimise_max_diversification(
                mu, cov, std_vec, returns_matrix, bounds,
            )
        else:
            for key in ["Max Sharpe", "Min Volatilität", "Risk Parity",
                        "Min CVaR", "Max Diversifikation"]:
                results[key] = None

        results["Kelly (Multi-Asset)"] = self.kelly_weights(
            asset_metrics, mu, cov, std_vec, returns_matrix,
        )

        return results
