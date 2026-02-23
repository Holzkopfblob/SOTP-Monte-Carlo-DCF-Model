"""
Excel Report Generator.

Creates a structured .xlsx workbook with:
  • Summary & Statistics    – Key metrics, percentiles, per-segment
  • Segment Assumptions     – Distribution configs used
  • Raw Simulation Data     – Full Monte-Carlo vectors for external audit

Uses **xlsxwriter** engine via pandas for maximum write performance.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from domain.models import (
    SimulationConfig,
    SimulationResults,
    TerminalValueMethod,
)


class ExcelExporter:
    """Generates a downloadable Excel report from simulation results."""

    def __init__(
        self,
        config: SimulationConfig,
        results: SimulationResults,
    ) -> None:
        self.config = config
        self.results = results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> bytes:
        """Return the Excel workbook as raw bytes (for st.download_button)."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            self._write_summary(writer)
            self._write_assumptions(writer)
            self._write_raw_data(writer)
            self._format_workbook(writer)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private sheets
    # ------------------------------------------------------------------

    def _write_summary(self, writer: pd.ExcelWriter) -> None:
        r = self.results
        rows: list[dict] = []

        def _row(label: str, arr: np.ndarray) -> dict:
            return {
                "Kennzahl": label,
                "Mittelwert": np.mean(arr),
                "Median": np.median(arr),
                "Std.-Abw.": np.std(arr),
                "P5": np.percentile(arr, 5),
                "P25": np.percentile(arr, 25),
                "P75": np.percentile(arr, 75),
                "P95": np.percentile(arr, 95),
                "Min": np.min(arr),
                "Max": np.max(arr),
            }

        rows.append(_row("Gesamt Enterprise Value", r.total_ev))
        for name, ev in r.segment_evs.items():
            rows.append(_row(f"  EV – {name}", ev))
        rows.append(_row("PV Holdingkosten", r.pv_corporate_costs))

        # Extended bridge items (as scalar rows using base-case values)
        if abs(r.base_minority_interests) > 0.01:
            rows.append({
                "Kennzahl": "Minderheitsanteile",
                **{k: -r.base_minority_interests for k in
                   ["Mittelwert", "Median", "Std.-Abw.", "P5", "P25", "P75", "P95", "Min", "Max"]},
            })
        if abs(r.base_pension_liabilities) > 0.01:
            rows.append({
                "Kennzahl": "Pensionsrückstellungen",
                **{k: -r.base_pension_liabilities for k in
                   ["Mittelwert", "Median", "Std.-Abw.", "P5", "P25", "P75", "P95", "Min", "Max"]},
            })
        if abs(r.base_non_operating_assets) > 0.01:
            rows.append({
                "Kennzahl": "Nicht-operative Assets",
                **{k: r.base_non_operating_assets for k in
                   ["Mittelwert", "Median", "Std.-Abw.", "P5", "P25", "P75", "P95", "Min", "Max"]},
            })
        if abs(r.base_associate_investments) > 0.01:
            rows.append({
                "Kennzahl": "Beteiligungen",
                **{k: r.base_associate_investments for k in
                   ["Mittelwert", "Median", "Std.-Abw.", "P5", "P25", "P75", "P95", "Min", "Max"]},
            })

        rows.append(_row("Equity Value", r.equity_values))
        rows.append(_row("Preis je Aktie", r.price_per_share))

        pd.DataFrame(rows).to_excel(
            writer, sheet_name="Summary & Statistics", index=False,
        )

    def _write_assumptions(self, writer: pd.ExcelWriter) -> None:
        rows: list[dict] = []
        for seg in self.config.segments:
            row: dict = {
                "Segment": seg.name,
                "Basisumsatz (Mio.)": seg.base_revenue,
                "Prognosejahre": seg.forecast_years,
                "TV-Methode": seg.terminal_method.value,
            }
            for attr, label in _PARAM_LABELS:
                dc = getattr(seg, attr)
                row[label] = dc.dist_type.value
            rows.append(row)
        pd.DataFrame(rows).to_excel(
            writer, sheet_name="Segment Assumptions", index=False,
        )

    def _write_raw_data(self, writer: pd.ExcelWriter) -> None:
        r = self.results
        data: dict[str, np.ndarray] = {}

        data["Simulation #"] = np.arange(1, r.n_simulations + 1)
        for k, v in r.input_samples.items():
            data[k] = v
        for k, v in r.segment_evs.items():
            data[f"EV – {k}"] = v
        data["Total EV"] = r.total_ev
        data["PV Holdingkosten"] = r.pv_corporate_costs
        data["Equity Value"] = r.equity_values
        data["Preis je Aktie"] = r.price_per_share

        # Cap at 100 000 rows to keep file size manageable
        max_rows = min(r.n_simulations, 100_000)
        df = pd.DataFrame({k: v[:max_rows] for k, v in data.items()})
        df.to_excel(writer, sheet_name="Raw Simulation Data", index=False)

    def _format_workbook(self, writer: pd.ExcelWriter) -> None:
        wb = writer.book
        num_fmt = wb.add_format({"num_format": "#,##0.00"})

        # Format Summary sheet
        ws = writer.sheets["Summary & Statistics"]
        ws.set_column("A:A", 30)
        ws.set_column("B:J", 18, num_fmt)

        # Format Assumptions sheet
        ws2 = writer.sheets["Segment Assumptions"]
        ws2.set_column("A:A", 25)
        ws2.set_column("B:Z", 22)


# ── Param-label mapping ──────────────────────────────────────────────────

_PARAM_LABELS = [
    ("revenue_growth",        "Umsatzwachstum"),
    ("ebitda_margin",         "EBITDA-Marge"),
    ("da_pct_revenue",        "D&A (% Umsatz)"),
    ("tax_rate",              "Steuersatz"),
    ("capex_pct_revenue",     "CAPEX (% Umsatz)"),
    ("nwc_pct_delta_revenue", "NWC (% ΔUmsatz)"),
    ("wacc",                  "WACC"),
    ("terminal_growth_rate",  "TV-Wachstum"),
    ("exit_multiple",         "Exit-Multiple"),
]
