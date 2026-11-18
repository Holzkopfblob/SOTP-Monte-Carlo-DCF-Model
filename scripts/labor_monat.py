"""Monatlicher Labor-Lauf (Nutzerentscheid 18.11.2026): SEC-Daten auffrischen, abgeschlossene Jahrgänge bewerten,
k und σ_x neu kalibrieren, nach Plausibilitätsprüfung in <SOTP_DATA>/labor_parameter.json übernehmen.

Abgeschlossen ist Jahrgang t, wenn der Börsenwert zum 30.06.t+5 gemeldet ist: das 10-K dazu kommt bis ~März t+6
⇒ ab April des Jahres J gilt t ≤ J − 6. Unvollständige Jahrgänge würden noch nicht meldende Firmen als Abgänge zählen.
Neue Werte nur, wenn plausibel (application.labor.GRENZEN); sonst bleiben die alten. Verlauf: labor_parameter_historie.csv.

Aufruf:  python scripts/labor_monat.py [--ohne-auffrischen]
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from application.labor import PARAMETER_DATEI, kalibriere, plausibel  # noqa: E402
from infrastructure.sec_xbrl import build_lab_panel, build_panel, data_dir  # noqa: E402
from scripts.labor_auswertung import faelle_mit_ergebnis  # noqa: E402


def letzter_jahrgang(heute: date) -> int:
    return heute.year - 6 if heute.month >= 4 else heute.year - 7


def auffrischen(heute: date) -> None:
    """Frames der letzten zwei Jahre neu holen (neue/korrigierte 10-K), Panels neu bauen (ältere Jahre aus dem Cache)."""
    d = data_dir()
    for f in (d / "frames").glob("*.parquet"):
        if any(f"CY{y}" in f.name for y in (heute.year - 2, heute.year - 1, heute.year)):
            f.unlink()
    alt = {n: d / n for n in ("panel.parquet", "labor_panel.parquet")}
    for n, f in alt.items():                  # sichern statt löschen: bei Fehler zurückspielen (App braucht das Panel)
        if f.exists():
            f.replace(d / (n + ".vorher"))
    try:
        build_panel().to_parquet(d / "panel.parquet")
        build_lab_panel()
    except Exception:
        for n, f in alt.items():
            if (d / (n + ".vorher")).exists():
                (d / (n + ".vorher")).replace(f)
        raise


def main() -> None:
    heute = date.today()
    t1 = letzter_jahrgang(heute)
    print(f"{heute}: abgeschlossene Jahrgänge 2014–{t1}", flush=True)
    if "--ohne-auffrischen" not in sys.argv:
        auffrischen(heute)
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "labor_run.py"), "--jahre", f"2014-{t1}"], check=True)
    p = kalibriere(faelle_mit_ergebnis(3), faelle_mit_ergebnis(5))
    p["stand"] = heute.isoformat()
    fehler = plausibel(p)
    f = data_dir() / PARAMETER_DATEI
    alt = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if fehler:
        print("NICHT übernommen (alte Werte bleiben):", "; ".join(fehler), flush=True)
    else:
        f.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
        print("übernommen:", {k: p[k] for k in ("k_vorsprung", "sigma_idio_5j", "idio_exponent")},
              "vorher:", {k: alt.get(k) for k in ("k_vorsprung", "sigma_idio_5j", "idio_exponent")}, flush=True)
    h = data_dir() / "labor_parameter_historie.csv"
    zeile = pd.DataFrame([{**p, "jahrgaenge": f"{min(p['jahrgaenge'])}–{max(p['jahrgaenge'])}",
                           "uebernommen": not fehler, "fehler": "; ".join(fehler)}])
    zeile.to_csv(h, mode="a", header=not h.exists(), index=False, sep=";")


if __name__ == "__main__":
    main()
