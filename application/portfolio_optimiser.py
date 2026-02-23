"""
Portfolio Optimisation Module.

Seven optimisation strategies: Max Sharpe, Min Vol, Risk Parity,
Min CVaR, Max Diversification, Multi-Asset Kelly, Equal Weight.
Also includes efficient frontier computation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from domain.portfolio_models import AssetMetrics, PortfolioResult


class PortfolioOptimiser:
    """Stateless portfolio optimisation engine.

    Instantiate with the risk-free rate, then call individual
    optimisation methods or :meth:`run_all`.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate

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
        bounds: list[tuple[float, float]] | None = None,
    ) -> PortfolioResult | None:
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
        bounds: list[tuple[float, float]] | None = None,
    ) -> PortfolioResult | None:
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
        bounds: list[tuple[float, float]] | None = None,
    ) -> PortfolioResult | None:
        """Equal Risk Contribution portfolio.

        Each asset contributes equally to total portfolio risk.
        User-defined bounds are respected (with a minimum floor of 0.01).
        """
        n = cov.shape[0]

        def risk_budget_obj(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = cov @ w
            risk_contrib = w * marginal / max(port_vol, 1e-12)
            target = port_vol / n
            return float(np.sum((risk_contrib - target) ** 2))

        w0 = np.ones(n) / n
        if bounds is None:
            bounds = [(0.01, 1.0)] * n
        else:
            bounds = [(max(lo, 0.01), hi) for lo, hi in bounds]
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
        bounds: list[tuple[float, float]] | None = None,
        alpha: float = 0.05,
    ) -> PortfolioResult | None:
        """Minimise Conditional Value at Risk (Expected Shortfall)."""
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
        bounds: list[tuple[float, float]] | None = None,
    ) -> PortfolioResult | None:
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
        asset_metrics: list[AssetMetrics],
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> PortfolioResult:
        """Multi-asset Kelly criterion (half-Kelly scaled)."""
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
        bounds: list[tuple[float, float]] | None = None,
        n_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute *n_points* on the efficient frontier.

        Returns ``(frontier_vols, frontier_rets)`` both as 1-D arrays.
        """
        n = len(mu)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n
        cons_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier_vols: list[float] = []
        frontier_rets: list[float] = []

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

    # ── Run all optimisations at once ─────────────────────────────────

    def run_all(
        self,
        asset_metrics: list[AssetMetrics],
        mu: np.ndarray,
        cov: np.ndarray,
        std_vec: np.ndarray,
        returns_matrix: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
    ) -> dict[str, PortfolioResult | None]:
        """Run all 7 optimisation strategies and return a name → result dict."""
        n = len(mu)
        results: dict[str, PortfolioResult | None] = {}

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
                mu, cov, std_vec, returns_matrix, bounds,
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
