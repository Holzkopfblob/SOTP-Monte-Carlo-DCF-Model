"""
Probability Distribution Abstraction Layer.

Provides an abstract interface (``BaseDistribution``) and concrete
implementations for Normal, LogNormal, Triangular, Uniform and PERT.
A factory function ``create_distribution`` maps a ``DistributionConfig``
to the correct concrete class.

All sampling is performed via **numpy's Generator API** for
reproducibility and maximum vectorised performance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from domain.models import DistributionConfig, DistributionType


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class BaseDistribution(ABC):
    """Interface that every probability distribution must implement."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw *n* independent samples.  Returns a 1-D array of shape (n,)."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a concise, human-readable description."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Concrete distributions
# ═══════════════════════════════════════════════════════════════════════════

class FixedDistribution(BaseDistribution):
    """Deterministic (constant) value – no randomness."""

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value, dtype=np.float64)

    def describe(self) -> str:
        return f"Fest: {self.value:.4f}"


class NormalDistribution(BaseDistribution):
    """Gaussian distribution  N(μ, σ²)."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = max(abs(std), 1e-12)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, size=n)

    def describe(self) -> str:
        return f"Normal(μ={self.mean:.4f}, σ={self.std:.4f})"


class LogNormalDistribution(BaseDistribution):
    """
    Log-normal distribution.

    Parameters *μ* and *σ* refer to the **underlying** normal distribution,
    i.e.  ln(X) ~ N(μ, σ²).
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = max(abs(sigma), 1e-12)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size=n)

    def describe(self) -> str:
        return f"LogNormal(μ={self.mu:.4f}, σ={self.sigma:.4f})"


class TriangularDistribution(BaseDistribution):
    """Triangular distribution  Tri(low, mode, high)."""

    def __init__(self, low: float, mode: float, high: float) -> None:
        self.low = low
        self.high = high if high > low else low + 1e-6
        self.mode = float(np.clip(mode, self.low, self.high))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def describe(self) -> str:
        return f"Dreieck(min={self.low:.4f}, mode={self.mode:.4f}, max={self.high:.4f})"


class UniformDistribution(BaseDistribution):
    """Continuous uniform distribution  U(low, high)."""

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high if high > low else low + 1e-6

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)

    def describe(self) -> str:
        return f"Gleichverteilung(min={self.low:.4f}, max={self.high:.4f})"


class PERTDistribution(BaseDistribution):
    """
    PERT (Program Evaluation and Review Technique) distribution.

    A re-scaled **Beta** distribution:

        α₁ = 1 + λ · (mode − low) / (high − low)
        α₂ = 1 + λ · (high − mode) / (high − low)
        X  = low + (high − low) · Beta(α₁, α₂)

    The default shape parameter λ = 4 gives the standard PERT
    distribution commonly used in financial expert-estimation.
    """

    def __init__(
        self,
        low: float,
        mode: float,
        high: float,
        lambd: float = 4.0,
    ) -> None:
        self.low = low
        self.high = high if high > low else low + 1e-6
        self.mode = float(np.clip(mode, self.low, self.high))
        self.lambd = lambd

        span = self.high - self.low
        self.alpha = 1.0 + lambd * (self.mode - self.low) / span
        self.beta_param = 1.0 + lambd * (self.high - self.mode) / span

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        beta_samples = rng.beta(self.alpha, self.beta_param, size=n)
        return self.low + (self.high - self.low) * beta_samples

    def describe(self) -> str:
        return (
            f"PERT(min={self.low:.4f}, mode={self.mode:.4f}, "
            f"max={self.high:.4f}, λ={self.lambd})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_distribution(config: DistributionConfig) -> BaseDistribution:
    """
    Factory function: map a ``DistributionConfig`` to the matching
    concrete ``BaseDistribution`` sub-class.
    """
    dt = config.dist_type

    if dt == DistributionType.FIXED:
        return FixedDistribution(config.fixed_value)

    if dt == DistributionType.NORMAL:
        return NormalDistribution(config.mean, config.std)

    if dt == DistributionType.LOGNORMAL:
        return LogNormalDistribution(config.ln_mu, config.ln_sigma)

    if dt == DistributionType.TRIANGULAR:
        return TriangularDistribution(config.low, config.mode, config.high)

    if dt == DistributionType.UNIFORM:
        return UniformDistribution(config.low, config.high)

    if dt == DistributionType.PERT:
        return PERTDistribution(config.low, config.mode, config.high)

    raise ValueError(f"Unbekannter Verteilungstyp: {dt}")
