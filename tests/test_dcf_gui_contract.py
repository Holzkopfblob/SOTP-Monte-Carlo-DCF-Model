"""GUI contract tests for the wizard-based DCF flow."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"
DCF_RESULTS = ROOT / "presentation" / "pages" / "dcf_results.py"


def _module_node(file_path: Path) -> ast.Module:
    return ast.parse(file_path.read_text(encoding="utf-8"))


def _function_node(file_path: Path, function_name: str) -> ast.FunctionDef:
    module = _module_node(file_path)
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


def _assignment_value(module: ast.Module, var_name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return node.value
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                return node.value
    raise AssertionError(f"Assignment for {var_name!r} not found")


def _is_streamlit_call(node: ast.Call, method_name: str) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "st"
        and func.attr == method_name
    )


def test_app_defines_wizard_steps_contract() -> None:
    """Wizard steps remain deterministic: setup -> segments -> simulation -> results."""
    module = _module_node(APP_FILE)
    wizard_steps = _assignment_value(module, "WIZARD_STEPS")
    assert isinstance(wizard_steps, ast.List)

    parsed_ids: list[str] = []
    for item in wizard_steps.elts:
        assert isinstance(item, ast.Tuple)
        assert len(item.elts) == 2
        step_id = item.elts[0]
        assert isinstance(step_id, ast.Constant)
        assert isinstance(step_id.value, str)
        parsed_ids.append(step_id.value)

    assert parsed_ids == ["setup", "segments", "simulation", "results"]


def test_app_no_longer_uses_tabs_navigation() -> None:
    """UI navigation contract: app shell uses wizard state, not st.tabs."""
    module = _module_node(APP_FILE)
    tabs_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _is_streamlit_call(node, "tabs")
    ]
    assert tabs_calls == []


def test_app_initializes_wizard_state_keys() -> None:
    """Wizard state keys are initialized in app shell session state."""
    module = _module_node(APP_FILE)
    source = APP_FILE.read_text(encoding="utf-8")
    assert "if \"wizard_step\" not in st.session_state" in source
    assert "if \"wizard_setup\" not in st.session_state" in source
    assert "if \"wizard_segments\" not in st.session_state" in source


def test_render_results_keeps_storyline_order() -> None:
    """Results storyline remains: executive summary -> risk -> drivers -> detail."""
    fn = _function_node(DCF_RESULTS, "render_results")
    calls = _called_render_functions(fn)

    executive_anchor = calls.index("render_key_metrics_section")
    risk_anchor = calls.index("render_tail_risk_section")
    driver_anchor = calls.index("render_sensitivity_section")
    detail_anchor = calls.index("render_descriptive_stats_section")

    assert executive_anchor < risk_anchor < driver_anchor < detail_anchor


def test_render_results_detail_flow_snapshot() -> None:
    """Detail flow remains deterministic for diagnostics and export placement."""
    fn = _function_node(DCF_RESULTS, "render_results")
    calls = _called_render_functions(fn)

    assert calls.index("render_descriptive_stats_section") < calls.index("render_distribution_section")
    assert calls.index("render_distribution_section") < calls.index("render_excel_export_section")
