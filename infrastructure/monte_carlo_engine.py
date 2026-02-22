"""
Vectorised Monte-Carlo Simulation Engine.

Orchestrates sampling of stochastic parameters and segment-level
DCF calculations.  The entire hot-path is **numpy-vectorised** –
no Python-level iteration over simulations.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from domain.distributions import create_distribution
from domain.models import (
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    SimulationResults,
    TerminalValueMethod,
)
from domain.valuation import compute_corporate_costs_pv, compute_segment_ev


class MonteCarloEngine:
    """Runs the full SOTP Monte-Carlo DCF simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResults:
        """Execute the simulation and return structured results."""
        n = self.config.n_simulations
        segment_evs: Dict[str, np.ndarray] = {}
        input_samples: Dict[str, np.ndarray] = {}

        for seg in self.config.segments:
            samples = self._sample_segment(seg, n)

            # Persist for sensitivity analysis
            for pname, arr in samples.items():
                input_samples[f"{seg.name} | {pname}"] = arr

            ev = compute_segment_ev(
                base_revenue=seg.base_revenue,
                forecast_years=seg.forecast_years,
                revenue_growth=samples["Umsatzwachstum"],
                ebitda_margin=samples["EBITDA-Marge"],
                da_pct_revenue=samples["D&A (% Umsatz)"],
                tax_rate=samples["Steuersatz"],
                capex_pct_revenue=samples["CAPEX (% Umsatz)"],
                nwc_pct_delta_revenue=samples["NWC (% ΔUmsatz)"],
                wacc=samples["WACC"],
                terminal_method=seg.terminal_method,
                terminal_growth=samples.get("TV-Wachstum", np.zeros(n)),
                exit_multiple=samples.get("Exit-Multiple", np.ones(n)),
                growth_mode=seg.revenue_growth_mode,
                fade_speed=seg.fade_speed,
            )
            segment_evs[seg.name] = ev

        # ── Aggregate ─────────────────────────────────────────────────
        total_ev = np.sum(
            np.column_stack(list(segment_evs.values())), axis=1
        )

        bridge = self.config.corporate_bridge

        # Sample stochastic corporate bridge parameters
        corp_costs_arr = self._sample_bridge_param(
            bridge.stochastic_corporate_costs,
            bridge.annual_corporate_costs, n,
        )
        net_debt_arr = self._sample_bridge_param(
            bridge.stochastic_net_debt,
            bridge.net_debt, n,
        )
        shares_arr = self._sample_bridge_param(
            bridge.stochastic_shares,
            bridge.shares_outstanding, n,
        )
        shares_arr = np.maximum(shares_arr, 1e-6)  # safety floor

        # Track stochastic bridge inputs for sensitivity
        if bridge.stochastic_corporate_costs is not None:
            input_samples["Bridge | Holdingkosten"] = corp_costs_arr
        if bridge.stochastic_net_debt is not None:
            input_samples["Bridge | Nettoverschuldung"] = net_debt_arr
        if bridge.stochastic_shares is not None:
            input_samples["Bridge | Aktien"] = shares_arr

        pv_corp = compute_corporate_costs_pv(
            corp_costs_arr,
            np.full(n, bridge.corporate_cost_discount_rate, dtype=np.float64),
        )
        equity_values = total_ev - pv_corp - net_debt_arr
        price_per_share = equity_values / shares_arr

        # Base-case means for the waterfall chart
        base_seg = {k: float(np.mean(v)) for k, v in segment_evs.items()}

        # ── Convergence diagnostics ───────────────────────────────────
        conv_idx, conv_mean, conv_lo, conv_hi = self._compute_convergence(
            equity_values
        )

        return SimulationResults(
            equity_values=equity_values,
            total_ev=total_ev,
            segment_evs=segment_evs,
            pv_corporate_costs=pv_corp,
            input_samples=input_samples,
            base_segment_evs=base_seg,
            base_corporate_costs_pv=float(np.mean(pv_corp)),
            base_net_debt=float(np.mean(net_debt_arr)),
            base_equity_value=float(np.mean(equity_values)),
            price_per_share=price_per_share,
            n_simulations=n,
            convergence_indices=conv_idx,
            convergence_means=conv_mean,
            convergence_ci_low=conv_lo,
            convergence_ci_high=conv_hi,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_bridge_param(
        self,
        dist_config,
        scalar_fallback: float,
        n: int,
    ) -> np.ndarray:
        """Sample a corporate bridge parameter – stochastic or deterministic."""
        if dist_config is not None:
            return create_distribution(dist_config).sample(n, self.rng)
        return np.full(n, scalar_fallback, dtype=np.float64)

    def _sample_segment(
        self, seg: SegmentConfig, n: int
    ) -> Dict[str, np.ndarray]:
        """
        Sample all stochastic parameters for a segment and apply
        domain-specific guards (e.g. WACC floor, TV-growth ceiling).
        """
        s: Dict[str, np.ndarray] = {}
        s["Umsatzwachstum"]   = create_distribution(seg.revenue_growth).sample(n, self.rng)
        s["EBITDA-Marge"]     = create_distribution(seg.ebitda_margin).sample(n, self.rng)
        s["D&A (% Umsatz)"]  = create_distribution(seg.da_pct_revenue).sample(n, self.rng)
        s["Steuersatz"]       = create_distribution(seg.tax_rate).sample(n, self.rng)
        s["CAPEX (% Umsatz)"] = create_distribution(seg.capex_pct_revenue).sample(n, self.rng)
        s["NWC (% ΔUmsatz)"]  = create_distribution(seg.nwc_pct_delta_revenue).sample(n, self.rng)
        s["WACC"]             = create_distribution(seg.wacc).sample(n, self.rng)

        # Guard: WACC must be > 0
        s["WACC"] = np.maximum(s["WACC"], 0.005)

        if seg.terminal_method == TerminalValueMethod.GORDON_GROWTH:
            s["TV-Wachstum"] = create_distribution(
                seg.terminal_growth_rate
            ).sample(n, self.rng)
            # Guard: terminal growth < WACC
            s["TV-Wachstum"] = np.minimum(
                s["TV-Wachstum"], s["WACC"] - 0.005
            )
        else:
            s["Exit-Multiple"] = create_distribution(
                seg.exit_multiple
            ).sample(n, self.rng)
            s["Exit-Multiple"] = np.maximum(s["Exit-Multiple"], 0.1)

        return s

    @staticmethod
    def _compute_convergence(
        equity_values: np.ndarray,
        n_checkpoints: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute running mean + 95 % CI at evenly spaced checkpoints.

        Returns (indices, running_means, ci_low, ci_high) – all (n_checkpoints,).
        """
        n = len(equity_values)
        if n < 20:
            idx = np.array([n])
            return idx, np.array([np.mean(equity_values)]), np.array([0.0]), np.array([0.0])

        checkpoints = np.unique(
            np.linspace(10, n, min(n_checkpoints, n), dtype=int)
        )

        means = np.empty(len(checkpoints))
        ci_lo = np.empty(len(checkpoints))
        ci_hi = np.empty(len(checkpoints))

        cumsum = np.cumsum(equity_values)
        cumsum_sq = np.cumsum(equity_values ** 2)

        for i, k in enumerate(checkpoints):
            m = cumsum[k - 1] / k
            var = cumsum_sq[k - 1] / k - m ** 2
            se = np.sqrt(max(var, 0.0) / k) * 1.96
            means[i] = m
            ci_lo[i] = m - se
            ci_hi[i] = m + se

        return checkpoints, means, ci_lo, ci_hi
