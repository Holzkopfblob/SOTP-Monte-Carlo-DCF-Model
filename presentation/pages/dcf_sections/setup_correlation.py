"""Cross-segment correlation section for DCF setup page."""

from __future__ import annotations

import numpy as np
import streamlit as st


def render_setup_correlation_section(n_segments: int) -> list[list[float]] | None:
    """Render cross-segment correlation controls and return matrix if valid."""
    st.subheader("🔗 Segment-Korrelation (Cross-Segment)")

    segment_correlation: list[list[float]] | None = None
    enable_corr = st.checkbox(
        "Segment-Korrelation aktivieren (Gauss-Copula)",
        value=False, key="setup_corr_enable",
        help="Fügt stochastische Abhängigkeit zwischen den Segmenten "
             "hinzu. Ein hoher Korrelationswert bedeutet, dass gute / "
             "schlechte Ergebnisse in verschiedenen Segmenten gemeinsam "
             "auftreten. Verwendet eine Gauss-Copula.",
    )

    if enable_corr and int(n_segments) >= 2:
        n_seg_int = int(n_segments)
        st.caption(
            f"Korrelationsmatrix ({n_seg_int}x{n_seg_int}) - Diagonal "
            "ist immer 1. Geben Sie die paarweisen Korrelationen "
            "zwischen den Segmenten ein (-1 bis 1)."
        )
        corr_values: list[list[float]] = [[1.0] * n_seg_int for _ in range(n_seg_int)]

        for row in range(n_seg_int):
            cols = st.columns(n_seg_int)
            for col_idx in range(n_seg_int):
                if col_idx == row:
                    cols[col_idx].number_input(
                        f"rho({row+1},{col_idx+1})",
                        value=1.0,
                        disabled=True,
                        key=f"corr_{row}_{col_idx}",
                    )
                elif col_idx > row:
                    val = cols[col_idx].number_input(
                        f"rho({row+1},{col_idx+1})",
                        value=0.3,
                        min_value=-1.0,
                        max_value=1.0,
                        step=0.05,
                        format="%.2f",
                        key=f"corr_{row}_{col_idx}",
                    )
                    corr_values[row][col_idx] = float(val)
                    corr_values[col_idx][row] = float(val)
                else:
                    cols[col_idx].number_input(
                        f"rho({row+1},{col_idx+1})",
                        value=float(corr_values[row][col_idx]),
                        disabled=True,
                        key=f"corr_{row}_{col_idx}",
                    )

        corr_arr = np.array(corr_values)
        eigvals = np.linalg.eigvalsh(corr_arr)
        if np.any(eigvals < -1e-8):
            st.warning(
                "⚠️ Die Matrix ist nicht positiv semi-definit. "
                "Bitte passen Sie die Korrelationswerte an."
            )
        else:
            segment_correlation = corr_values

    elif enable_corr and int(n_segments) < 2:
        st.info("Korrelation erfordert mindestens 2 Segmente.")

    return segment_correlation
