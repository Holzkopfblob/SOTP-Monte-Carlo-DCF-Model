"""
Portfolio Service – Facade Module
==================================

Thin composition layer that wires together the three sub-modules
(:mod:`portfolio_analyser`, :mod:`portfolio_optimiser`,
:mod:`portfolio_stress`) and exposes the original ``PortfolioAnalyser``
API so that **all existing imports continue to work unchanged**.

Internal structure (new)::

    application/
        portfolio_analyser.py   – AssetAnalyser   (single-asset metrics)
        portfolio_optimiser.py  – PortfolioOptimiser  (7 optimisation methods)
        portfolio_stress.py     – PortfolioStressTester (stress scenarios)
        portfolio_service.py    – PortfolioAnalyser (facade, this file)

The ``generate_fv_samples`` function and the sector-correlation utilities
remain here because they don't fit neatly into the three sub-modules.
"""
from __future__ import annotations

import numpy as np

from domain.portfolio_models import (
    AssetInput,
    AssetMetrics,
    CovarianceMethod,
    InvestorView,
    PortfolioResult,
    StressTestResult,
)
from domain.distributions import create_distribution
from domain.models import DistributionConfig, DistributionType

# Sub-modules
from application.portfolio_analyser import AssetAnalyser
from application.portfolio_optimiser import PortfolioOptimiser
from application.portfolio_stress import PortfolioStressTester

# Re-export so existing imports keep working
__all__ = [
    "AssetInput",
    "AssetMetrics",
    "CovarianceMethod",
    "InvestorView",
    "PortfolioResult",
    "StressTestResult",
    "PortfolioAnalyser",
    "AssetAnalyser",
    "PortfolioOptimiser",
    "PortfolioStressTester",
    "generate_fv_samples",
    "ledoit_wolf_shrinkage",
    "SECTOR_CLUSTERS",
    "SAME_SECTOR_CORR",
    "DEFAULT_CROSS_CORR",
]


# ═══════════════════════════════════════════════════════════════════════════
# Sector correlation model
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_CLUSTERS: dict[str, str] = {
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

_CLUSTER_CORR: dict[tuple[str, str], float] = {
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
    return _CLUSTER_CORR.get(
        (c1, c2),
        _CLUSTER_CORR.get((c2, c1), DEFAULT_CROSS_CORR),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fair-value sample generation
# ═══════════════════════════════════════════════════════════════════════════

def _to_lognormal_params(desired_mean: float, desired_std: float):
    """Convert desired (mean, std) of a lognormal RV to underlying normal (μ, σ)."""
    if desired_mean > 0 and desired_std > 0:
        sigma_sq = np.log(1.0 + (desired_std / desired_mean) ** 2)
        mu_ln = np.log(desired_mean) - sigma_sq / 2.0
        sigma_ln = np.sqrt(sigma_sq)
        return mu_ln, sigma_ln
    return np.log(max(desired_mean, 1e-12)), 0.1


def _map_portfolio_dist(dist_type: str, params: dict) -> DistributionConfig:
    """Map a portfolio-app distribution name + params to a DistributionConfig."""
    if dist_type == "Normal":
        return DistributionConfig(
            dist_type=DistributionType.NORMAL,
            mean=params["mean"], std=params["std"],
        )
    if dist_type == "Lognormal":
        mu_ln, sigma_ln = _to_lognormal_params(params["mean"], params["std"])
        return DistributionConfig(
            dist_type=DistributionType.LOGNORMAL,
            ln_mu=mu_ln, ln_sigma=sigma_ln,
        )
    if dist_type == "Aus DCF-App (μ, σ, Schiefe)":
        mu, sigma = params["mean"], params["std"]
        skewness = params.get("skew", 0.0)
        if abs(skewness) < 0.5:
            return DistributionConfig(
                dist_type=DistributionType.NORMAL,
                mean=mu, std=sigma,
            )
        mu_ln, sigma_ln = _to_lognormal_params(mu, sigma)
        return DistributionConfig(
            dist_type=DistributionType.LOGNORMAL,
            ln_mu=mu_ln, ln_sigma=sigma_ln,
        )
    if dist_type == "PERT":
        return DistributionConfig(
            dist_type=DistributionType.PERT,
            low=params["low"], mode=params["mode"], high=params["high"],
        )
    if dist_type == "Dreiecksverteilung":
        return DistributionConfig(
            dist_type=DistributionType.TRIANGULAR,
            low=params["low"], mode=params["mode"], high=params["high"],
        )
    if dist_type == "Gleichverteilung":
        return DistributionConfig(
            dist_type=DistributionType.UNIFORM,
            low=params["low"], high=params["high"],
        )
    return DistributionConfig(
        dist_type=DistributionType.FIXED,
        fixed_value=params.get("mean", 0.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Ledoit-Wolf Shrinkage Estimator
# ═══════════════════════════════════════════════════════════════════════════

def ledoit_wolf_shrinkage(
    returns_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) analytical linear shrinkage estimator.

    Shrinks the sample covariance matrix toward a scaled identity
    (constant-variance) target.  This reduces estimation error in
    the covariance matrix, especially when *p* (assets) is large
    relative to *n* (observations).

    Returns ``(shrunk_cov, shrinkage_intensity)`` where intensity ∈ [0, 1].
    """
    n, p = returns_matrix.shape
    if p < 2 or n < 2:
        S = np.cov(returns_matrix, rowvar=False, ddof=1)
        if S.ndim == 0:
            S = np.array([[float(S)]])
        return S, 0.0

    X = returns_matrix - returns_matrix.mean(axis=0)
    S = (X.T @ X) / n                      # biased sample cov

    # Target: scaled identity  F = μ · I  where μ = trace(S) / p
    mu_target = np.trace(S) / p
    delta_mat = S - mu_target * np.eye(p)

    # Squared Frobenius norms
    delta_sq_sum = np.sum(delta_mat ** 2)   # ||S - F||²_F

    # Compute φ̂² (numerator of optimal shrinkage)
    # φ̂² = (1/n²) Σ_k ||x_k x_k' - S||²_F
    Y = X ** 2
    phi = np.sum((Y.T @ Y) / n - S ** 2)

    # Optimal shrinkage
    kappa = phi / max(delta_sq_sum, 1e-18)
    shrinkage = max(0.0, min(1.0, kappa / n))

    S_shrunk = (1.0 - shrinkage) * S + shrinkage * mu_target * np.eye(p)
    return S_shrunk, float(shrinkage)


def generate_fv_samples(
    dist_type: str,
    params: dict,
    n: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Generate *n* fair-value Monte-Carlo samples from the chosen distribution.

    Returns an (n,) array clipped to a minimum of 0.01.
    """
    rng = np.random.default_rng(seed)
    config = _map_portfolio_dist(dist_type, params)
    samples = create_distribution(config).sample(n, rng)
    return np.maximum(samples, 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Facade – composes AssetAnalyser + PortfolioOptimiser + PortfolioStressTester
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioAnalyser:
    """Backward-compatible facade that delegates to the focused sub-modules.

    Every existing call site continues to work unchanged.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate
        self._analyser = AssetAnalyser(risk_free_rate)
        self._optimiser = PortfolioOptimiser(risk_free_rate)
        self._stress = PortfolioStressTester(risk_free_rate)

    # ── Single-asset analysis (→ AssetAnalyser) ──────────────────────

    def analyse_asset(self, asset: AssetInput) -> AssetMetrics:
        return self._analyser.analyse_asset(asset)

    def analyse_all(self, assets: list[AssetInput]) -> list[AssetMetrics]:
        return self._analyser.analyse_all(assets)

    # ── Matrix construction (kept here – utility-level) ───────────────

    @staticmethod
    def build_returns_matrix(assets: list[AssetInput]) -> np.ndarray:
        """Stack per-asset return vectors into an (n_sim, n_assets) matrix."""
        return np.column_stack([
            (a.fv_samples / a.current_price) - 1.0 for a in assets
        ])

    @staticmethod
    def build_sector_correlation(sectors: list[str]) -> np.ndarray:
        """Create a correlation matrix from sector assignments."""
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
        """Project a symmetric matrix to the nearest PSD correlation matrix."""
        return PortfolioStressTester.ensure_psd(matrix, epsilon)

    @staticmethod
    def build_covariance(
        returns_matrix: np.ndarray,
        correlation_matrix: np.ndarray | None = None,
        method: CovarianceMethod = CovarianceMethod.SAMPLE,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (mu, std, cov) from returns.

        When *method* is ``LEDOIT_WOLF``, the covariance is computed
        via the Ledoit-Wolf shrinkage estimator (ignoring
        *correlation_matrix*).
        """
        mu = np.mean(returns_matrix, axis=0)
        stds = np.std(returns_matrix, axis=0)

        if method == CovarianceMethod.LEDOIT_WOLF:
            cov, _ = ledoit_wolf_shrinkage(returns_matrix)
            return mu, stds, cov

        if correlation_matrix is not None:
            corr = correlation_matrix
        else:
            corr = np.corrcoef(returns_matrix, rowvar=False)
            if corr.ndim == 0:
                corr = np.array([[1.0]])
        cov = np.outer(stds, stds) * corr
        return mu, stds, cov

    # ── Optimisation (→ PortfolioOptimiser) ───────────────────────────

    def optimise_max_sharpe(self, mu, cov, std_vec, returns_matrix, bounds=None):
        return self._optimiser.optimise_max_sharpe(mu, cov, std_vec, returns_matrix, bounds)

    def optimise_min_vol(self, mu, cov, std_vec, returns_matrix, bounds=None):
        return self._optimiser.optimise_min_vol(mu, cov, std_vec, returns_matrix, bounds)

    def optimise_risk_parity(self, mu, cov, std_vec, returns_matrix, bounds=None):
        return self._optimiser.optimise_risk_parity(mu, cov, std_vec, returns_matrix, bounds)

    def optimise_min_cvar(self, mu, cov, std_vec, returns_matrix, bounds=None, alpha=0.05):
        return self._optimiser.optimise_min_cvar(mu, cov, std_vec, returns_matrix, bounds, alpha)

    def optimise_max_diversification(self, mu, cov, std_vec, returns_matrix, bounds=None):
        return self._optimiser.optimise_max_diversification(mu, cov, std_vec, returns_matrix, bounds)

    def kelly_weights(self, asset_metrics, mu, cov, std_vec, returns_matrix):
        return self._optimiser.kelly_weights(asset_metrics, mu, cov, std_vec, returns_matrix)

    def equal_weights(self, n, mu, cov, std_vec, returns_matrix):
        return self._optimiser.equal_weights(n, mu, cov, std_vec, returns_matrix)

    def efficient_frontier(self, mu, cov, bounds=None, n_points=50):
        return self._optimiser.efficient_frontier(mu, cov, bounds, n_points)

    def run_all_optimisations(self, asset_metrics, mu, cov, std_vec, returns_matrix, bounds=None):
        """Backward-compatible alias for ``PortfolioOptimiser.run_all``."""
        return self._optimiser.run_all(asset_metrics, mu, cov, std_vec, returns_matrix, bounds)

    def optimise_hrp(self, mu, cov, std_vec, returns_matrix):
        return self._optimiser.optimise_hrp(mu, cov, std_vec, returns_matrix)

    def black_litterman(self, mu_mkt, cov, std_vec, returns_matrix, views,
                        tau=0.05, bounds=None):
        return self._optimiser.black_litterman(
            mu_mkt, cov, std_vec, returns_matrix, views, tau, bounds,
        )

    # ── Stress testing (→ PortfolioStressTester) ──────────────────────

    def stress_test(self, portfolios, returns_matrix, asset_sectors,
                    market_shock_pct=-30.0, corr_stress=0.85,
                    sector_shock=None, sector_shock_pct=0.0):
        return self._stress.stress_test(
            portfolios, returns_matrix, asset_sectors,
            market_shock_pct, corr_stress, sector_shock, sector_shock_pct,
        )
