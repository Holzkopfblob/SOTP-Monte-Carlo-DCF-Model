"""
DCF Versionsvergleich – gespeicherte Bewertungen über die Zeit
===============================================================
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from infrastructure.config_io import diff_configs
from presentation.charts import valuation_timeline

_METRICS = {
    "preis_je_aktie": "Preis je Aktie",
    "equity_value": "Equity Value (Mio.)",
    "enterprise_value": "Enterprise Value (Mio.)",
}


def _load_versions(files) -> list[tuple[str, dict]]:
    """Parse uploaded files into (unique label, config), oldest first."""
    versions: list[tuple[str, dict]] = []
    for f in files:
        try:
            cfg = json.loads(f.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            st.error(f"{f.name}: keine gültige JSON-Datei – übersprungen.")
            continue
        company = (cfg.get("metadata") or {}).get("company") or f.name.rsplit(".", 1)[0]
        stamp = str(cfg.get("saved_at", ""))[:16].replace("T", " ")
        versions.append((f"{company} · {stamp}" if stamp else company, cfg, f.name))
    versions.sort(key=lambda v: str(v[1].get("saved_at", "")))

    out: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for label, cfg, name in versions:
        if label in seen:
            label = f"{label} ({name})"
        seen.add(label)
        out.append((label, cfg))
    return out


def render_compare(container) -> None:
    """Render the version comparison view (independent of the wizard)."""
    with container:
        st.header("🔍 Versionsvergleich")
        st.caption(
            "Jede gespeicherte Konfiguration ist ein Zeitpunkt: Annahmen, Quellen "
            "und – wenn vor dem Speichern simuliert wurde – das Ergebnis. "
            "Laden Sie zwei oder mehr Dateien derselben Bewertung."
        )
        files = st.file_uploader(
            "Gespeicherte Konfigurationen (.json)", type=["json"],
            accept_multiple_files=True, key="compare_upload",
        )
        versions = _load_versions(files or [])
        if len(versions) < 2:
            st.info("Mindestens zwei Dateien hochladen, um zu vergleichen.")
            return
        cfgs = dict(versions)
        labels = list(cfgs)

        # ── Results over time ─────────────────────────────────────────
        st.subheader("📈 Ergebnis über die Zeit")
        with_result = [(lbl, c) for lbl, c in versions if c.get("ergebnis")]
        if not with_result:
            st.info(
                "Keine Datei enthält ein Ergebnis. Führen Sie die Simulation vor dem "
                "Speichern aus – dann werden P5/P50/P95 automatisch mitgespeichert."
            )
        else:
            metric = st.radio(
                "Kennzahl", list(_METRICS), format_func=_METRICS.get,
                horizontal=True, key="compare_metric",
            )
            stats = [c["ergebnis"][metric] for _, c in with_result]
            prices = (
                [(c.get("metadata") or {}).get("kurs_bei_recherche") for _, c in with_result]
                if metric == "preis_je_aktie" else None
            )
            st.plotly_chart(
                valuation_timeline(
                    [lbl for lbl, _ in with_result],
                    [s["p5"] for s in stats], [s["p50"] for s in stats], [s["p95"] for s in stats],
                    market_prices=prices, y_label=_METRICS[metric],
                ),
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(
                    [{"Version": lbl, "P5": s["p5"], "P50": s["p50"], "P95": s["p95"], "Ø": s["mean"]}
                     for (lbl, _), s in zip(with_result, stats)]
                ).style.format({k: "{:,.2f}" for k in ("P5", "P50", "P95", "Ø")}),
                width="stretch", hide_index=True,
            )
            if len(with_result) < len(versions):
                st.caption(f"{len(versions) - len(with_result)} Datei(en) ohne gespeichertes Ergebnis nicht im Chart.")

        # ── Assumption diff ───────────────────────────────────────────
        st.subheader("🧾 Was hat sich an den Annahmen geändert?")
        c1, c2 = st.columns(2)
        old_lbl = c1.selectbox("Alt", labels, index=len(labels) - 2, key="compare_old")
        new_lbl = c2.selectbox("Neu", labels, index=len(labels) - 1, key="compare_new")
        rows = diff_configs(cfgs[old_lbl], cfgs[new_lbl])
        if not rows:
            st.success("Keine Unterschiede in den Annahmen.")
        else:
            df = pd.DataFrame(rows).rename(columns={
                "bereich": "Bereich", "parameter": "Parameter",
                "alt": "Alt", "neu": "Neu", "quelle": "Begründung (neu)",
            })
            df[["Alt", "Neu"]] = df[["Alt", "Neu"]].astype(str)
            st.caption(f"{len(rows)} geänderte Eingaben.")
            st.dataframe(df, width="stretch", hide_index=True)

        verz = cfgs[new_lbl].get("quellenverzeichnis")
        if verz:
            with st.expander("📚 Quellenverzeichnis der neuen Version"):
                for key, ref in verz.items():
                    st.markdown(f"**[{key}]** {ref}")
