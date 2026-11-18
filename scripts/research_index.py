"""Research-Index: hält die Firmen-Dossiers aktuell und listet fällige Termine.

Ablage (Vault, $SOTP_RESEARCH): firmen.csv (Ticker = ID), fakten.csv (Ist-Zahlen je Quartal mit Quelle),
termine.csv (Nachverfolgung), Firmen/<TICKER>.md (These, Auslöser, Entscheidungslog). Dieses Skript schreibt in
jedem Dossier nur die Blöcke zwischen den Markern <!-- historie:start/end --> und <!-- termine:start/end -->
(aus annahmen_register.csv bzw. termine.csv); alles andere im Dossier gehört dem Menschen.
Für gehaltene Titel ohne Dossier legt es ein Gerüst an.

Aufruf:  python scripts/research_index.py [--tage 21]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

FIN = Path(r"C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen")
RESEARCH = Path(os.environ.get("SOTP_RESEARCH", FIN / "SOTP-Research"))
REGISTER = Path(os.environ.get("SOTP_BEWERTUNGEN", FIN / "SOTP-Monte-Carlo-DCF_Bewertungen")) / "annahmen_register.csv"
DEPOT = Path(os.environ.get("SOTP_DEPOT", FIN / "depot.json"))
HOME = Path.home()

GERUEST = """# {firma} ({ticker})

Sektor {sektor} · Gruppe {gruppe} · Währung {waehrung} · Analyseordner `{analyseordner}`

## These
_Warum unterbewertet? Was muss eintreten, damit der Markt es sieht?_

## Auslöser
- Nachkaufen, wenn …
- Verkaufen, wenn … (These widerlegt, nicht Kurs gefallen)

## Bewertungshistorie
<!-- historie:start -->
<!-- historie:end -->

## Termine
<!-- termine:start -->
<!-- termine:end -->

## Entscheidungslog
| Datum | Entscheidung | Grund |
|---|---|---|

## Offene Punkte
"""


def _read(path: Path, delim: str) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def historie(register: list[dict], firma: str) -> str:
    """Eine Zeile je Bewertungsversion (Firma, Datum, Stand) mit Kurs, P5/P50/P95, Lücke und IRR-P50."""
    versions: dict[tuple, dict] = {}
    for r in register:
        if r["firma"] != firma:
            continue
        v = versions.setdefault((r["datum"], r["stand"]), {"kurs": r["kurs"], "p5": r["p5"], "p50": r["p50"],
                                                           "p95": r["p95"], "irr": None})
        if r.get("bereich") == "IRR":
            v["irr"] = _num(r.get("mode"))
    if not versions:
        return "_Noch keine Bewertung im Register._"
    lines = ["| Datum | Stand | Kurs | P5 / P50 / P95 | P50/Kurs | IRR P50 |", "|---|---|---|---|---|---|"]
    for (dt, stand), v in sorted(versions.items()):
        k, p50 = _num(v["kurs"]), _num(v["p50"])
        gap = f"{p50 / k - 1:+.0%}" if k and p50 else ""
        irr = f"{v['irr']:.1f} %" if v["irr"] is not None else ""
        lines.append(f"| {dt} | {stand} | {v['kurs']} | {_fmt(v['p5'])} / {_fmt(v['p50'])} / {_fmt(v['p95'])} | {gap} | {irr} |")
    return "\n".join(lines)


def _fmt(x: str) -> str:
    n = _num(x)
    return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if n is not None else ""


def termine_block(termine: list[dict], ticker: str) -> str:
    rows = [t for t in termine if t["ticker"] == ticker and t["status"] != "erledigt"]
    if not rows:
        return "_Keine offenen Termine._"
    return "\n".join(f"- {t['datum']} · {t['ereignis']}: {t['aktion']} ({t['status']})" for t in sorted(rows, key=lambda t: t["datum"]))


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text) if pat.search(text) else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tage", type=int, default=21)
    a = ap.parse_args()
    firmen = _read(RESEARCH / "firmen.csv", ";")
    termine = _read(RESEARCH / "termine.csv", ";")
    register = _read(REGISTER, ",")
    held = {p["ticker"] for p in json.loads(DEPOT.read_text(encoding="utf-8"))["positionen"]
            if float(p.get("stueck") or 0) > 0} if DEPOT.exists() else set()
    (RESEARCH / "Firmen").mkdir(parents=True, exist_ok=True)

    for f in firmen:
        path = RESEARCH / "Firmen" / f"{f['ticker']}.md"
        if not path.exists() and f["ticker"] not in held:
            continue
        text = path.read_text(encoding="utf-8") if path.exists() else GERUEST.format(**f)
        text = replace_block(text, "historie", historie(register, f["firma"]))
        text = replace_block(text, "termine", termine_block(termine, f["ticker"]))
        path.write_text(text, encoding="utf-8")
        if f.get("analyseordner") and not (HOME / f["analyseordner"]).exists():
            print(f"⚠ {f['ticker']}: Analyseordner fehlt: {f['analyseordner']}")

    known = {f["ticker"] for f in firmen}
    for t in held - known:
        print(f"⚠ {t}: im Depot gehalten, fehlt in firmen.csv")
    today, horizon = date.today(), date.today() + timedelta(days=a.tage)
    due = sorted((t for t in termine if t["status"] != "erledigt" and date.fromisoformat(t["datum"]) <= horizon),
                 key=lambda t: t["datum"])
    print(f"Termine bis {horizon} ({len(due)}):")
    for t in due:
        tag = "ÜBERFÄLLIG " if date.fromisoformat(t["datum"]) < today else ""
        print(f"  {tag}{t['datum']} {t['ticker']:8} {t['ereignis']}: {t['aktion']}")


if __name__ == "__main__":
    main()
