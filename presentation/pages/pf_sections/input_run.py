"""Run and config-save section for portfolio input."""
from __future__ import annotations

import json

import numpy as np
import streamlit as st

from application.portfolio_service import (
    AssetInput,
    CovarianceMethod,
    InvestorView,
    PortfolioAnalyser,
    generate_fv_samples,
)


def render_input_run_section(
    *,
    asset_configs: list[dict],
    corr_matrix: np.ndarray,
    corr_method: str,
    sectors: list[str],
    cov_method: CovarianceMethod,
    n_mc_sim: int,
    global_seed: int,
    risk_free_pct: float,
    enable_bl: bool,
    bl_views: list[dict],
) -> None:
    """Render save-and-run controls and execute portfolio analysis."""
    st.divider()

    save_col, run_col = st.columns([1, 2])
    with save_col:
        config_json = json.dumps({
            "assets": asset_configs,
            "corr_method": corr_method,
            "risk_free_pct": risk_free_pct,
            "n_mc_sim": int(n_mc_sim),
            "seed": int(global_seed),
        }, indent=2, default=str)
        st.download_button(
            "💾 Konfiguration speichern",
            data=config_json,
            file_name="portfolio_config.json",
            mime="application/json",
        )

    with run_col:
        run_analysis = st.button(
            "🚀 Portfolio-Analyse starten",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        n_total = len(asset_configs)
        with st.spinner("Generiere Fair-Value-Verteilungen & optimiere Portfolio …"):
            asset_inputs: list[AssetInput] = []
            for idx, ac in enumerate(asset_configs):
                fv = generate_fv_samples(
                    ac["dist_type"], ac["params"],
                    n=int(n_mc_sim),
                    seed=int(global_seed) + idx,
                )
                asset_inputs.append(AssetInput(
                    name=ac["name"],
                    sector=ac["sector"],
                    current_price=ac["price"],
                    fv_samples=fv,
                    min_weight=ac["min_weight"],
                    max_weight=ac["max_weight"],
                ))

            analyser = PortfolioAnalyser(risk_free_rate=risk_free_pct / 100.0)
            asset_metrics = analyser.analyse_all(asset_inputs)
            returns_matrix = analyser.build_returns_matrix(asset_inputs)
            mu_vec, std_vec, cov_matrix = analyser.build_covariance(
                returns_matrix,
                corr_matrix if cov_method == CovarianceMethod.SAMPLE else None,
                method=cov_method,
            )
            bounds = [(ai.min_weight, ai.max_weight) for ai in asset_inputs]
            opt_results = analyser.run_all_optimisations(
                asset_metrics, mu_vec, cov_matrix, std_vec, returns_matrix, bounds,
            )

            if enable_bl and bl_views:
                asset_names = [ac["name"] for ac in asset_configs]
                view_objects = [
                    InvestorView(
                        asset_index=asset_names.index(v["asset_name"]),
                        expected_return=v["expected_return"],
                        confidence=v["confidence"],
                    )
                    for v in bl_views
                ]
                bl_result = analyser.black_litterman(
                    mu_vec, cov_matrix, std_vec, returns_matrix,
                    view_objects, bounds=bounds,
                )
                if bl_result is not None:
                    opt_results["Black-Litterman"] = bl_result

            if n_total >= 2:
                ef_vols, ef_rets = analyser.efficient_frontier(
                    mu_vec, cov_matrix, bounds,
                )
            else:
                ef_vols, ef_rets = np.array([]), np.array([])

            st.session_state.pf_results = {
                "asset_metrics": asset_metrics,
                "asset_inputs": asset_inputs,
                "corr_matrix": corr_matrix,
                "cov_matrix": cov_matrix,
                "mu_vec": mu_vec,
                "std_vec": std_vec,
                "returns_matrix": returns_matrix,
                "rf": risk_free_pct / 100.0,
                "opt_results": opt_results,
                "ef_vols": ef_vols,
                "ef_rets": ef_rets,
                "n_sim": int(n_mc_sim),
                "sectors": sectors,
            }

        st.success(
            f"✅ Analyse abgeschlossen – {n_total} Assets · "
            f"{int(n_mc_sim):,} MC-Simulationen · "
            f"{sum(1 for v in opt_results.values() if v is not None)} Optimierungsmethoden"
        )
        st.balloons()
