"""
The Streamlit pages and the headless builder must produce the same config.

Renders the real setup and segments pages (streamlit.testing.AppTest) for
the bundled Airbus example and compares the result with
``build_simulation_config``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from infrastructure.config_io import build_simulation_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "airbus_sotp_config.json"


def _page():
    import json
    import os
    import pickle
    import streamlit as st
    from infrastructure.config_io import apply_config
    from presentation.pages.dcf_segments import render_segments
    from presentation.pages.dcf_setup import render_setup

    if "_loaded" not in st.session_state:
        cfg = json.load(open(os.environ["SOTP_TEST_CFG"], encoding="utf-8"))
        for k, v in apply_config(cfg, {}).items():
            st.session_state[k] = v
        st.session_state["_loaded"] = True
    setup = render_setup(st.container())
    segs = render_segments(st.container(), int(setup["n_segments"]))
    with open(os.environ["SOTP_TEST_OUT"], "wb") as f:
        pickle.dump({"setup": setup, "segments": segs}, f)


def test_pages_match_builder(tmp_path, monkeypatch):
    out = tmp_path / "ui.pkl"
    monkeypatch.setenv("SOTP_TEST_CFG", str(CFG))
    monkeypatch.setenv("SOTP_TEST_OUT", str(out))
    at = AppTest.from_function(_page, default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    ui = pickle.loads(out.read_bytes())
    built = build_simulation_config(json.loads(CFG.read_text(encoding="utf-8")))
    assert ui["segments"] == built.segments
    assert ui["setup"]["segment_correlation"] == built.segment_correlation
    assert ui["setup"]["n_simulations"] == built.n_simulations
