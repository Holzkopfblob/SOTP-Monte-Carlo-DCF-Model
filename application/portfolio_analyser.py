"""
Single-Asset Analysis Module.

Computes 12+ metrics per asset from MC fair-value distributions.
"""
from __future__ import annotations

import numpy as np

from domain.portfolio_models import AssetInput, AssetMetrics


class AssetAnalyser:
    """Stateless single-asset analysis engine.

    Instantiate with the risk-free rate, then call :meth:`analyse_asset`
    or :meth:`analyse_all`.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate

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

    def analyse_all(self, assets: list[AssetInput]) -> list[AssetMetrics]:
        """Analyse every asset and return a list of metrics."""
        return [self.analyse_asset(a) for a in assets]
