"""
Portfolio Stress-Testing Module.

Supports market shocks, correlation stress (Cholesky re-correlation),
sector-specific crisis scenarios, historical scenarios, and macro
factor sensitivity.
"""
from __future__ import annotations

import numpy as np

from domain.portfolio_models import (
    HistoricalScenario,
    MACRO_SECTOR_SENSITIVITY,
    StressTestResult,
)


class PortfolioStressTester:
    """Stateless stress-testing engine for portfolio weight vectors."""

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate

    @staticmethod
    def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Project a symmetric matrix to the nearest PSD matrix
        and re-normalise to a valid correlation matrix (diag = 1).
        """
        B = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, epsilon)
        result = eigvecs @ np.diag(eigvals) @ eigvecs.T
        result = (result + result.T) / 2.0
        d = np.sqrt(np.diag(result))
        if np.all(d > 0):
            result = result / np.outer(d, d)
            np.fill_diagonal(result, 1.0)
        return result

    def stress_test(
        self,
        portfolios: dict[str, np.ndarray],
        returns_matrix: np.ndarray,
        asset_sectors: list[str],
        market_shock_pct: float = -30.0,
        corr_stress: float = 0.85,
        sector_shock: str | None = None,
        sector_shock_pct: float = 0.0,
    ) -> tuple[list[StressTestResult], np.ndarray]:
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

        Returns
        -------
        results : list of StressTestResult
        shocked_returns : (n_sim, n_assets) array
        """
        shocked_returns = returns_matrix + (market_shock_pct / 100.0)

        if sector_shock and sector_shock_pct != 0:
            for idx, sec in enumerate(asset_sectors):
                if sec == sector_shock:
                    shocked_returns[:, idx] += (sector_shock_pct / 100.0)

        # ── Correlation stress via Cholesky re-correlation ────────────
        if corr_stress > 0 and shocked_returns.shape[1] > 1:
            n_assets = shocked_returns.shape[1]
            mu_s = np.mean(shocked_returns, axis=0)
            std_s = np.std(shocked_returns, axis=0)
            std_safe = np.maximum(std_s, 1e-12)

            z = (shocked_returns - mu_s) / std_safe

            corr_orig = np.corrcoef(z, rowvar=False)
            if corr_orig.ndim == 0:
                corr_orig = np.array([[1.0]])
            corr_orig = self.ensure_psd(corr_orig)

            corr_new = corr_orig.copy()
            mask = ~np.eye(n_assets, dtype=bool)
            corr_new[mask] = np.maximum(corr_orig[mask], corr_stress)
            corr_new = self.ensure_psd(corr_new)

            try:
                L_orig = np.linalg.cholesky(corr_orig)
                L_new = np.linalg.cholesky(corr_new)
                z_indep = np.linalg.solve(L_orig, z.T).T
                z_stressed = (L_new @ z_indep.T).T
                shocked_returns = z_stressed * std_safe + mu_s
            except np.linalg.LinAlgError:
                pass  # fall back to unmodified shocked returns

        results: list[StressTestResult] = []
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

        return results, shocked_returns

    # ── Historical scenario shortcut ──────────────────────────────────

    def stress_test_scenario(
        self,
        scenario: HistoricalScenario,
        portfolios: dict[str, np.ndarray],
        returns_matrix: np.ndarray,
        asset_sectors: list[str],
    ) -> tuple[list[StressTestResult], np.ndarray]:
        """Apply a :class:`HistoricalScenario` as a stress test.

        The scenario's sector-specific shocks are applied on top of
        the general market shock.
        """
        # Aggregate sector shocks into per-asset additional shock
        shocked_returns = returns_matrix + (scenario.market_shock_pct / 100.0)
        for idx, sec in enumerate(asset_sectors):
            extra = scenario.sector_shocks.get(sec, 0.0)
            if extra != 0:
                shocked_returns[:, idx] += (extra / 100.0)

        # Correlation stress (same logic as normal stress_test)
        if scenario.corr_stress > 0 and shocked_returns.shape[1] > 1:
            n_assets = shocked_returns.shape[1]
            mu_s = np.mean(shocked_returns, axis=0)
            std_s = np.std(shocked_returns, axis=0)
            std_safe = np.maximum(std_s, 1e-12)
            z = (shocked_returns - mu_s) / std_safe

            corr_orig = np.corrcoef(z, rowvar=False)
            if corr_orig.ndim == 0:
                corr_orig = np.array([[1.0]])
            corr_orig = self.ensure_psd(corr_orig)

            corr_new = corr_orig.copy()
            mask = ~np.eye(n_assets, dtype=bool)
            corr_new[mask] = np.maximum(corr_orig[mask], scenario.corr_stress)
            corr_new = self.ensure_psd(corr_new)

            try:
                L_orig = np.linalg.cholesky(corr_orig)
                L_new = np.linalg.cholesky(corr_new)
                z_indep = np.linalg.solve(L_orig, z.T).T
                z_stressed = (L_new @ z_indep.T).T
                shocked_returns = z_stressed * std_safe + mu_s
            except np.linalg.LinAlgError:
                pass

        results: list[StressTestResult] = []
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

        return results, shocked_returns

    # ── Macro factor sensitivity ──────────────────────────────────────

    @staticmethod
    def macro_factor_impact(
        asset_sectors: list[str],
        interest_rate_delta: float = 0.0,
        inflation_delta: float = 0.0,
        gdp_delta: float = 0.0,
    ) -> np.ndarray:
        """Estimate per-asset return delta from macro factor changes.

        Uses the sector-specific sensitivity table from
        ``MACRO_SECTOR_SENSITIVITY``.

        Parameters
        ----------
        asset_sectors : list of sector names (one per asset)
        interest_rate_delta : change in long-term interest rates (pp)
        inflation_delta : change in inflation expectations (pp)
        gdp_delta : change in real GDP growth (pp)

        Returns
        -------
        (n_assets,) array of estimated return impact (decimal).
        """
        impacts = np.zeros(len(asset_sectors))
        for i, sector in enumerate(asset_sectors):
            sens = MACRO_SECTOR_SENSITIVITY.get(sector, {})
            delta = (
                sens.get("Zinsen", 0.0) * interest_rate_delta
                + sens.get("Inflation", 0.0) * inflation_delta
                + sens.get("BIP", 0.0) * gdp_delta
            )
            impacts[i] = delta / 100.0  # convert % → decimal
        return impacts
