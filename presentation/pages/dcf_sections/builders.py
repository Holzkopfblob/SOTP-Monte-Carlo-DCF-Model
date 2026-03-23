"""Data builders for DCF result sections."""

from __future__ import annotations

import pandas as pd

from domain.statistics import compute_statistics


def build_key_metrics(results) -> dict[str, dict[str, float]]:
    """Build key statistics for EV, Equity and Price-per-share arrays."""
    return {
        "ev": compute_statistics(results.total_ev),
        "equity": compute_statistics(results.equity_values),
        "price": compute_statistics(results.price_per_share),
    }


def build_stats_table(results) -> pd.DataFrame:
    """Build descriptive statistics table including per-segment EV stats."""
    key_metrics = build_key_metrics(results)
    stats_data: dict[str, dict[str, float]] = {
        "Enterprise Value": key_metrics["ev"],
        "Equity Value": key_metrics["equity"],
        "Preis / Aktie": key_metrics["price"],
    }
    for seg_name, seg_ev in results.segment_evs.items():
        stats_data[f"EV – {seg_name}"] = compute_statistics(seg_ev)
    return pd.DataFrame(stats_data).T


def build_tail_risk_metrics(results) -> dict[str, float | None]:
    """Build tail-risk metric payload used by risk sections."""
    return {
        "var_5": getattr(results, "equity_var_5", None),
        "cvar_5": getattr(results, "equity_cvar_5", None),
        "tail_ratio": getattr(results, "equity_tail_ratio", None),
    }


def build_quality_payload(results) -> dict[str, object]:
    """Build payload for quality score section."""
    return {
        "score": getattr(results, "quality_score", None),
        "tv_ev_ratios": getattr(results, "segment_tv_ev_ratios", None),
    }
