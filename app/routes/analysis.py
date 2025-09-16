"""Analysis page route for NeuroVoxel multipage UI."""

import streamlit as st

from app.features.data_loader import analysis_form


class AnalysisPage:
    """Page for running neuroimaging analysis."""

    @staticmethod
    def render() -> None:
        """Render the analysis form and handle user input."""
        st.header("Run Analysis")
        dataset_path, analysis_type, run_clicked = analysis_form()
        if run_clicked:
            st.write(f"Running {analysis_type} analysis on {dataset_path}")
