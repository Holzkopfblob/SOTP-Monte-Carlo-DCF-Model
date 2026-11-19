"""Export section for DCF results."""

from __future__ import annotations

import streamlit as st

from infrastructure.excel_export import ExcelExporter


def render_excel_export_section(config, results) -> None:
    """Render Excel report download section."""
    st.subheader("📥 Excel-Export")
    st.caption(
        "Drei Arbeitsblätter: Summary & Statistics, Segment Assumptions und Raw Simulation Data "
        "(alle Szenarien, daher erst auf Knopfdruck erzeugt)."
    )

    # Building the workbook (all raw scenarios) is slow – only on demand, once per result.
    cache = st.session_state.get("_excel_cache")
    if cache is None or cache[0] is not results:
        if st.button("📄 Excel-Report erzeugen", width="stretch"):
            with st.spinner("Erzeuge Excel-Report …"):
                st.session_state["_excel_cache"] = cache = (results, ExcelExporter(config, results).generate())
        else:
            return
    excel_bytes = cache[1]
    st.download_button(
        label="📥 Vollständigen Excel-Report herunterladen",
        data=excel_bytes,
        file_name="sotp_mc_dcf_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        type="primary",
        width="stretch",
    )
