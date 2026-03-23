"""Covariance method section for portfolio setup."""
from __future__ import annotations

import streamlit as st

from application.portfolio_service import CovarianceMethod


def render_input_covariance_section() -> tuple[str, CovarianceMethod]:
    """Render covariance method selector and return label plus enum."""
    st.divider()
    st.subheader("📊 Kovarianzschätzung")

    cov_method_label = st.radio(
        "Methode zur Kovarianzschätzung",
        [CovarianceMethod.SAMPLE.value, CovarianceMethod.LEDOIT_WOLF.value],
        horizontal=True,
        help="Ledoit-Wolf schrumpft die Sample-Kovarianz zum Identitäts-Target "
             "und reduziert Schätzfehler — besonders bei vielen Assets.",
    )
    cov_method = (
        CovarianceMethod.LEDOIT_WOLF
        if cov_method_label == CovarianceMethod.LEDOIT_WOLF.value
        else CovarianceMethod.SAMPLE
    )
    return cov_method_label, cov_method
