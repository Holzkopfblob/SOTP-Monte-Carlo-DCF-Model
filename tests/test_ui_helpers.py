"""
Tests for presentation.ui_helpers – distribution input renderer & exports.

For Streamlit widget functions we can only test the non-UI parts.
"""
from __future__ import annotations

import pytest

from presentation.layout import states
from presentation.ui_helpers import DIST_OPTIONS, render_info_box
from domain.models import DistributionType


class TestDistOptions:
    def test_all_types_covered(self):
        """DIST_OPTIONS must contain a label for every DistributionType."""
        for dt in DistributionType:
            assert dt.value in DIST_OPTIONS, f"{dt.value} missing from DIST_OPTIONS"

    def test_no_duplicates(self):
        assert len(DIST_OPTIONS) == len(set(DIST_OPTIONS))


class _DummyExpander:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestInfoRegistry:
    def test_render_info_box_uses_registry_text(self, monkeypatch):
        calls: list[str] = []

        monkeypatch.setattr("presentation.ui_helpers.st.expander", lambda _title: _DummyExpander())
        monkeypatch.setattr("presentation.ui_helpers.st.markdown", lambda text: calls.append(text))

        render_info_box("fcff")

        assert len(calls) == 1
        assert "Free Cash Flow to Firm" in calls[0]


class TestLayoutStates:
    @pytest.mark.parametrize(
        ("fn_name", "message", "expected"),
        [
            ("render_empty_state", "Leerer Zustand", "info"),
            ("render_warning_state", "Warnung", "warning"),
            ("render_success_state", "Erfolg", "success"),
        ],
    )
    def test_state_renderers_delegate_to_streamlit(self, monkeypatch, fn_name, message, expected):
        event_log: list[tuple[str, str]] = []

        monkeypatch.setattr(states.st, "subheader", lambda text: event_log.append(("subheader", text)))
        monkeypatch.setattr(states.st, "info", lambda text: event_log.append(("info", text)))
        monkeypatch.setattr(states.st, "warning", lambda text: event_log.append(("warning", text)))
        monkeypatch.setattr(states.st, "success", lambda text: event_log.append(("success", text)))

        fn = getattr(states, fn_name)
        fn(message, title="Titel")

        assert event_log[0] == ("subheader", "Titel")
        assert event_log[1] == (expected, message)
