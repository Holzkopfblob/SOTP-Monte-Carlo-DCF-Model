"""Export section for DCF results."""

from __future__ import annotations

import streamlit as st

from infrastructure.excel_export import ExcelExporter


def render_excel_export_section(config, results) -> None:
    """Render Excel report download section."""
    st.subheader("📥 Excel-Export")
    st.markdown(
        "Der Report enthält drei Arbeitsblätter: "
        "**Summary & Statistics**, **Segment Assumptions** "
        "und **Raw Simulation Data**."
    )

    excel_bytes = ExcelExporter(config, results).generate()
    st.download_button(
        label="📥 Vollständigen Excel-Report herunterladen",
        data=excel_bytes,
        file_name="sotp_mc_dcf_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True,
    )
