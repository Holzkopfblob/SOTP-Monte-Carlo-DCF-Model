"""GUI contract tests for DCF setup/results orchestrators."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DCF_SETUP = ROOT / "presentation" / "pages" / "dcf_setup.py"
DCF_RESULTS = ROOT / "presentation" / "pages" / "dcf_results.py"

EXPECTED_SETUP_KEYS = {
    "n_simulations",
    "random_seed",
    "n_segments",
    "mid_year_conv",
    "sampling_method",
    "bridge_corp_costs",
    "bridge_corp_discount",
    "bridge_net_debt",
    "bridge_shares",
    "bridge_minority",
    "bridge_pension",
    "bridge_non_op",
    "bridge_associates",
    "segment_correlation",
}


def _function_node(file_path: Path, function_name: str) -> ast.FunctionDef:
    module = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Function {function_name!r} not found in {file_path}")


def _called_render_functions(func: ast.FunctionDef) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id.startswith("render_"):
                calls.append(node.func.id)
    return calls


def _return_dict_keys(func: ast.FunctionDef) -> set[str]:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys: set[str] = set()
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
            return keys
    raise AssertionError("No dict return found")


def test_render_setup_returns_stable_contract_keys() -> None:
    """render_setup must keep the stable key contract used by simulation/config IO."""
    fn = _function_node(DCF_SETUP, "render_setup")
    assert _return_dict_keys(fn) == EXPECTED_SETUP_KEYS


def test_render_results_enforces_insight_order_snapshot() -> None:
    """render_results follows the agreed insight order: Executive -> Risiko -> Treiber -> Detail."""
    fn = _function_node(DCF_RESULTS, "render_results")
    calls = _called_render_functions(fn)

    executive_anchor = calls.index("render_key_metrics_section")
    risk_anchor = calls.index("render_tail_risk_section")
    driver_anchor = calls.index("render_sensitivity_section")
    detail_anchor = calls.index("render_descriptive_stats_section")

    assert executive_anchor < risk_anchor < driver_anchor < detail_anchor


def test_render_results_detail_flow_snapshot() -> None:
    """Detail flow remains deterministic for diagnostics, segment detail and exports."""
    fn = _function_node(DCF_RESULTS, "render_results")
    calls = _called_render_functions(fn)

    assert calls.index("render_descriptive_stats_section") < calls.index("render_distribution_section")
    assert calls.index("render_distribution_section") < calls.index("render_portfolio_handoff_section")
    assert calls.index("render_portfolio_handoff_section") < calls.index("render_excel_export_section")
