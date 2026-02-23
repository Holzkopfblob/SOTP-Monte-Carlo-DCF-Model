"""
Vectorised Monte-Carlo Simulation Engine.

Orchestrates sampling of stochastic parameters and segment-level
DCF calculations.  The entire hot-path is **numpy-vectorised** –
no Python-level iteration over simulations.
"""
from __future__ import annotations


import numpy as np
from scipy import stats as sp_stats

from domain.distributions import create_distribution
from domain.fade import build_fade_curve
from domain.models import (
    DistributionConfig,
    RevenueGrowthMode,
    SegmentConfig,
    SimulationConfig,
    SimulationResults,
    TerminalValueMethod,
)
from domain.valuation import compute_corporate_costs_pv, compute_segment_ev
from domain.valuation_metrics import (
    implied_roic,
    reinvestment_rate,
    tv_ev_ratio,
    valuation_quality_score,
)


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
        segment_evs: dict[str, np.ndarray] = {}
        segment_tv_ev: dict[str, np.ndarray] = {}
        segment_roic: dict[str, np.ndarray] = {}
        segment_reinvest: dict[str, np.ndarray] = {}
        input_samples: dict[str, np.ndarray] = {}

        # ── Phase 3: choose sampling strategy ─────────────────────────
        corr = self.config.segment_correlation
        if corr is not None and len(self.config.segments) > 1:
            corr_mat = np.asarray(corr, dtype=np.float64)
            all_samples = self._sample_all_segments_correlated(
                self.config.segments, corr_mat, n,
            )
        else:
            all_samples = [
                self._sample_segment(seg, n) for seg in self.config.segments
            ]

        for seg, samples in zip(self.config.segments, all_samples):
            # Persist for sensitivity analysis – use initial (Year-1)
            # value when a param is time-varying (n, T)
            for pname, arr in samples.items():
                if arr.ndim == 2:
                    input_samples[f"{seg.name} | {pname}"] = arr[:, 0]
                else:
                    input_samples[f"{seg.name} | {pname}"] = arr

            detail = compute_segment_ev(
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
                mid_year_convention=self.config.mid_year_convention,
                decompose=True,
            )
            segment_evs[seg.name] = detail.ev

            # ── Phase 2 metrics ───────────────────────────────────────
            segment_tv_ev[seg.name] = tv_ev_ratio(detail.pv_tv, detail.ev)

            # implied ROIC / reinvestment use (n,) scalars; when fade
            # yields (n, T), take the Year-1 column as representative.
            def _y1(arr: np.ndarray) -> np.ndarray:
                return arr[:, 0] if arr.ndim == 2 else arr

            segment_roic[seg.name] = implied_roic(
                _y1(samples["EBITDA-Marge"]),
                _y1(samples["D&A (% Umsatz)"]),
                _y1(samples["Steuersatz"]),
                _y1(samples["CAPEX (% Umsatz)"]),
                _y1(samples["NWC (% ΔUmsatz)"]),
                samples["Umsatzwachstum"],
            )
            segment_reinvest[seg.name] = reinvestment_rate(
                _y1(samples["CAPEX (% Umsatz)"]),
                _y1(samples["D&A (% Umsatz)"]),
                _y1(samples["NWC (% ΔUmsatz)"]),
                samples["Umsatzwachstum"],
                _y1(samples["EBITDA-Marge"]),
                _y1(samples["Steuersatz"]),
            )

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
        corp_disc_arr = self._sample_bridge_param(
            bridge.stochastic_corporate_cost_discount_rate,
            bridge.corporate_cost_discount_rate, n,
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

        # ── Extended bridge items ─────────────────────────────────────
        minority_arr = self._sample_bridge_param(
            bridge.stochastic_minority_interests,
            bridge.minority_interests, n,
        )
        pension_arr = self._sample_bridge_param(
            bridge.stochastic_pension_liabilities,
            bridge.pension_liabilities, n,
        )
        non_op_arr = self._sample_bridge_param(
            bridge.stochastic_non_operating_assets,
            bridge.non_operating_assets, n,
        )
        associates_arr = self._sample_bridge_param(
            bridge.stochastic_associate_investments,
            bridge.associate_investments, n,
        )

        # Track stochastic bridge inputs for sensitivity
        if bridge.stochastic_corporate_costs is not None:
            input_samples["Bridge | Holdingkosten"] = corp_costs_arr
        if bridge.stochastic_corporate_cost_discount_rate is not None:
            input_samples["Bridge | Diskontierung Holding"] = corp_disc_arr
        if bridge.stochastic_net_debt is not None:
            input_samples["Bridge | Nettoverschuldung"] = net_debt_arr
        if bridge.stochastic_shares is not None:
            input_samples["Bridge | Aktien"] = shares_arr
        if bridge.stochastic_minority_interests is not None:
            input_samples["Bridge | Minderheitsanteile"] = minority_arr
        if bridge.stochastic_pension_liabilities is not None:
            input_samples["Bridge | Pensionsrückstellungen"] = pension_arr
        if bridge.stochastic_non_operating_assets is not None:
            input_samples["Bridge | Nicht-operative Assets"] = non_op_arr
        if bridge.stochastic_associate_investments is not None:
            input_samples["Bridge | Beteiligungen"] = associates_arr

        pv_corp = compute_corporate_costs_pv(
            corp_costs_arr,
            corp_disc_arr,
        )
        # Extended equity bridge:
        # Equity = Sum(EV) - PV(Corp) - NetDebt - Minority - Pension
        #          + NonOperating + Associates
        equity_values = (
            total_ev
            - pv_corp
            - net_debt_arr
            - minority_arr
            - pension_arr
            + non_op_arr
            + associates_arr
        )
        price_per_share = equity_values / shares_arr

        # Base-case means for the waterfall chart
        base_seg = {k: float(np.mean(v)) for k, v in segment_evs.items()}

        # ── Convergence diagnostics ───────────────────────────────────
        conv_idx, conv_mean, conv_lo, conv_hi = self._compute_convergence(
            equity_values
        )

        # ── Phase 2: Quality score ────────────────────────────────────
        # Aggregate TV/EV across segments (weighted by mean EV)
        total_mean_ev = float(np.mean(total_ev))
        if total_mean_ev > 1e-6:
            weighted_tv_ev = sum(
                float(np.mean(segment_tv_ev[s])) * float(np.mean(segment_evs[s]))
                for s in segment_evs
            ) / total_mean_ev
        else:
            weighted_tv_ev = 0.0

        ci_width_pct = 0.0
        if len(conv_hi) > 0 and abs(conv_mean[-1]) > 0:
            ci_width_pct = float(
                (conv_hi[-1] - conv_lo[-1]) / abs(conv_mean[-1]) * 100
            )

        # Sensitivity (will be computed externally, pass empty for now
        # and let the service layer fill it in after construction)
        from application.simulation_service import SimulationService
        _tmp_results = SimulationResults(
            equity_values=equity_values,
            total_ev=total_ev,
            segment_evs=segment_evs,
            pv_corporate_costs=pv_corp,
            input_samples=input_samples,
        )
        sensitivities = SimulationService.compute_sensitivity(_tmp_results)

        q_score = valuation_quality_score(
            mean_tv_ev=weighted_tv_ev,
            ci_width_pct=ci_width_pct,
            correlations=sensitivities,
            equity_mean=float(np.mean(equity_values)),
            equity_std=float(np.std(equity_values)),
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
            base_minority_interests=float(np.mean(minority_arr)),
            base_pension_liabilities=float(np.mean(pension_arr)),
            base_non_operating_assets=float(np.mean(non_op_arr)),
            base_associate_investments=float(np.mean(associates_arr)),
            price_per_share=price_per_share,
            n_simulations=n,
            convergence_indices=conv_idx,
            convergence_means=conv_mean,
            convergence_ci_low=conv_lo,
            convergence_ci_high=conv_hi,
            segment_tv_ev_ratios=segment_tv_ev,
            segment_implied_roic=segment_roic,
            segment_reinvest_rates=segment_reinvest,
            quality_score=q_score,
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

    # ── Parameter fade (Phase 3) ──────────────────────────────────────

    def _apply_param_fade(
        self,
        initial: np.ndarray,
        terminal_cfg: DistributionConfig | None,
        n: int,
        forecast_years: int,
        fade_speed: float,
    ) -> np.ndarray:
        """Return (n,) array when no fade or (n, T) fade curve.

        If *terminal_cfg* is ``None`` the initial (n,) array passes
        through unchanged.  Otherwise the terminal values are sampled
        and an exponentially-decaying path is built via
        ``build_fade_curve``.
        """
        if terminal_cfg is None:
            return initial  # (n,) – constant across years
        terminal = create_distribution(terminal_cfg).sample(n, self.rng)
        return build_fade_curve(initial, terminal, forecast_years, fade_speed)  # (n, T)

    # ── Segment sampling ──────────────────────────────────────────────

    def _sample_segment(
        self, seg: SegmentConfig, n: int
    ) -> dict[str, np.ndarray]:
        """
        Sample all stochastic parameters for a segment and apply
        domain-specific guards (e.g. WACC floor, TV-growth ceiling).
        When a terminal DistributionConfig is set for a param, the
        parameter fades from sampling-initial → terminal over *T* years.
        """
        s: dict[str, np.ndarray] = {}
        s["Umsatzwachstum"]   = create_distribution(seg.revenue_growth).sample(n, self.rng)

        # Value drivers – (n,) or (n, T) when fade is active
        s["EBITDA-Marge"]     = self._apply_param_fade(
            create_distribution(seg.ebitda_margin).sample(n, self.rng),
            seg.ebitda_margin_terminal, n, seg.forecast_years, seg.fade_speed,
        )
        s["D&A (% Umsatz)"]  = self._apply_param_fade(
            create_distribution(seg.da_pct_revenue).sample(n, self.rng),
            seg.da_pct_revenue_terminal, n, seg.forecast_years, seg.fade_speed,
        )
        s["Steuersatz"]       = self._apply_param_fade(
            create_distribution(seg.tax_rate).sample(n, self.rng),
            seg.tax_rate_terminal, n, seg.forecast_years, seg.fade_speed,
        )
        s["CAPEX (% Umsatz)"] = self._apply_param_fade(
            create_distribution(seg.capex_pct_revenue).sample(n, self.rng),
            seg.capex_pct_revenue_terminal, n, seg.forecast_years, seg.fade_speed,
        )
        s["NWC (% ΔUmsatz)"]  = self._apply_param_fade(
            create_distribution(seg.nwc_pct_delta_revenue).sample(n, self.rng),
            seg.nwc_pct_delta_revenue_terminal, n, seg.forecast_years, seg.fade_speed,
        )
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

    # ── Gaussian copula for cross-segment correlation (Phase 3) ───────

    def _sample_all_segments_correlated(
        self,
        segments: list[SegmentConfig],
        corr_matrix: np.ndarray,
        n: int,
    ) -> list[dict[str, np.ndarray]]:
        """Sample every segment using a Gaussian copula.

        Instead of sampling each segment independently, we:
        1. Draw *m* correlated standard normals via Cholesky (one per segment).
        2. Transform to uniform marginals via Φ.
        3. Use the inverse CDF (ppf) of each parameter's distribution to
           map the uniform draws back to the desired marginal.

        The same inter-segment correlation drives **all** stochastic params
        within a segment, preserving each param's marginal distribution while
        introducing dependence *between* segments.
        """
        m = len(segments)
        L = np.linalg.cholesky(corr_matrix)   # (m, m)

        # Draw independent standard normals  →  (m, n)
        z_indep = self.rng.standard_normal(size=(m, n))
        # Apply Cholesky to correlate  →  (m, n)
        z_corr = L @ z_indep
        # Map to uniform [0, 1] via Φ
        u_corr = sp_stats.norm.cdf(z_corr)    # (m, n)

        all_samples: list[dict[str, np.ndarray]] = []

        for idx, seg in enumerate(segments):
            u = u_corr[idx]  # (n,) uniform marginals for this segment

            s: dict[str, np.ndarray] = {}

            # Revenue growth – always use copula for initial draw
            s["Umsatzwachstum"] = create_distribution(seg.revenue_growth).ppf(u)

            # Value drivers – copula for initial, then optionally fade
            _PARAM_MAP = [
                ("EBITDA-Marge",     seg.ebitda_margin,         seg.ebitda_margin_terminal),
                ("D&A (% Umsatz)",  seg.da_pct_revenue,        seg.da_pct_revenue_terminal),
                ("Steuersatz",       seg.tax_rate,              seg.tax_rate_terminal),
                ("CAPEX (% Umsatz)", seg.capex_pct_revenue,    seg.capex_pct_revenue_terminal),
                ("NWC (% ΔUmsatz)",  seg.nwc_pct_delta_revenue, seg.nwc_pct_delta_revenue_terminal),
            ]
            for pname, init_cfg, term_cfg in _PARAM_MAP:
                initial = create_distribution(init_cfg).ppf(u)
                s[pname] = self._apply_param_fade(
                    initial, term_cfg, n, seg.forecast_years, seg.fade_speed,
                )

            s["WACC"] = create_distribution(seg.wacc).ppf(u)
            s["WACC"] = np.maximum(s["WACC"], 0.005)

            if seg.terminal_method == TerminalValueMethod.GORDON_GROWTH:
                s["TV-Wachstum"] = create_distribution(
                    seg.terminal_growth_rate
                ).ppf(u)
                s["TV-Wachstum"] = np.minimum(
                    s["TV-Wachstum"], s["WACC"] - 0.005
                )
            else:
                s["Exit-Multiple"] = create_distribution(
                    seg.exit_multiple
                ).ppf(u)
                s["Exit-Multiple"] = np.maximum(s["Exit-Multiple"], 0.1)

            all_samples.append(s)

        return all_samples

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
