"""Research-Index: schreibt nur die markierten Blöcke im Dossier, der Rest gehört dem Menschen."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ri", Path(__file__).parents[1] / "scripts" / "research_index.py")
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)


def test_nur_markierte_bloecke_werden_ersetzt():
    text = "# X\nThese bleibt\n<!-- historie:start -->\nalt\n<!-- historie:end -->\nLog bleibt\n"
    out = ri.replace_block(text, "historie", "neu")
    assert "neu" in out and "alt" not in out and "These bleibt" in out and "Log bleibt" in out
    assert ri.replace_block("ohne Marker", "historie", "neu") == "ohne Marker"


def test_historie_eine_zeile_je_version_mit_irr():
    reg = [{"firma": "A", "datum": "2026-11-06", "stand": "vor", "kurs": "100", "p5": "90", "p50": "150", "p95": "200",
            "bereich": "Setup", "mode": ""},
           {"firma": "A", "datum": "2026-11-06", "stand": "vor", "kurs": "100", "p5": "90", "p50": "150", "p95": "200",
            "bereich": "IRR", "mode": "12.5"},
           {"firma": "B", "datum": "2026-11-06", "stand": "x", "kurs": "1", "p5": "1", "p50": "1", "p95": "1",
            "bereich": "Setup", "mode": ""}]
    h = ri.historie(reg, "A")
    assert h.count("\n| 2026") == 1 and "+50%" in h and "12.5 %" in h
