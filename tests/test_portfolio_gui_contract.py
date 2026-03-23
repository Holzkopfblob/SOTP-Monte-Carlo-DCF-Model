"""GUI contract tests for portfolio orchestrators and shared constants."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_APP = ROOT / "portfolio_app.py"
PF_COMMON = ROOT / "presentation" / "pages" / "pf_common.py"
PF_INPUT = ROOT / "presentation" / "pages" / "pf_input.py"

EXPECTED_TAB_ORDER = [
    "📝 Bewertungen eingeben",
    "🔍 Einzeltitel-Analyse",
    "📊 Portfolio-Optimierung",
    "📈 Efficient Frontier",
    "⚡ Stress-Tests",
]

EXPECTED_RENDER_ORDER = [
    "render_input",
    "render_single",
    "render_portfolio",
    "render_frontier",
    "render_stress",
]

EXPECTED_METHOD_ORDER = [
    "Gleichgewicht (1/N)",
    "Max Sharpe",
    "Min Volatilität",
    "Risk Parity",
    "Min CVaR",
    "Max Diversifikation",
    "Kelly (Multi-Asset)",
    "HRP",
    "Black-Litterman",
]

EXPECTED_SESSION_STATE_KEYS = {
    "pf_results",
    "_pf_loaded_cfg",
}


def _parse(file_path: Path) -> ast.Module:
    return ast.parse(file_path.read_text(encoding="utf-8"))


def _extract_tabs(module: ast.Module) -> list[str]:
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "tabs" and node.args and isinstance(node.args[0], ast.List):
                out: list[str] = []
                for elt in node.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.append(elt.value)
                return out
    raise AssertionError("st.tabs(...) call not found")


def _extract_call_sequence(module: ast.Module, names: list[str]) -> list[str]:
    calls: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id in names:
                calls.append(call.func.id)
    return calls


def _extract_list_assignment(module: ast.Module, variable_name: str) -> list[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
                if isinstance(node.value, ast.List):
                    values: list[str] = []
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            values.append(elt.value)
                    return values
    raise AssertionError(f"List assignment for {variable_name!r} not found")


def _extract_session_state_keys(module: ast.Module) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                if isinstance(node.value.value, ast.Name) and node.value.value.id == "st":
                    if node.value.attr == "session_state":
                        keys.add(node.attr)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "st":
                if node.value.attr == "session_state" and isinstance(node.slice, ast.Constant):
                    if isinstance(node.slice.value, str):
                        keys.add(node.slice.value)
    return keys


def test_portfolio_app_tab_order_contract() -> None:
    """Portfolio app keeps the agreed tab order for navigation consistency."""
    module = _parse(PORTFOLIO_APP)
    assert _extract_tabs(module) == EXPECTED_TAB_ORDER


def test_portfolio_app_orchestrator_render_order_contract() -> None:
    """Thin orchestrator must render section modules in deterministic order."""
    module = _parse(PORTFOLIO_APP)
    calls = _extract_call_sequence(module, EXPECTED_RENDER_ORDER)
    assert calls == EXPECTED_RENDER_ORDER


def test_method_order_contract_snapshot() -> None:
    """METHOD_ORDER remains stable for all portfolio presentation modules."""
    module = _parse(PF_COMMON)
    assert _extract_list_assignment(module, "METHOD_ORDER") == EXPECTED_METHOD_ORDER


def test_portfolio_session_state_key_contract() -> None:
    """Session state keys used by portfolio flow remain stable."""
    app_keys = _extract_session_state_keys(_parse(PORTFOLIO_APP))
    input_keys = _extract_session_state_keys(_parse(PF_INPUT))
    assert EXPECTED_SESSION_STATE_KEYS.issubset(app_keys | input_keys)
