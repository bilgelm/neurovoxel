"""Visualization feature for the NeuroVoxel app.

This module provides the UI components for visualizing results using Streamlit.
"""

import streamlit as st
from pandas import Index

from neurovoxel.utils.viz import (
    basic_interactive_viz,  # pyright: ignore[reportUnknownVariableType]
    nanslice_overlay,
)


def render_visualization(inference_terms: Index) -> None:
    """Render the visualization UI for NeuroVoxel app."""
    for variable in inference_terms:  # pyright: ignore[reportUnknownVariableType, reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
        html_view = basic_interactive_viz(
            st.session_state.result,
            st.session_state.masker,
            term=variable,
            stat="t",
            bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
            opacity=0.5,  # pyright: ignore[reportArgumentType]
            title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
        )
        html_content = html_view.get_iframe()  # pyright: ignore[reportUnknownMemberType]
        st.components.v1.html(html_content, width=650, height=250)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        p_thresh = 0.01
        fig = nanslice_overlay(
            st.session_state.result,
            st.session_state.masker,
            bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
            term=variable,
            p_var="logp_max_t",
            p_thresh=p_thresh,
            title=(
                "Regression coefficient overlay with "
                f"corrected p < {p_thresh} contours for {variable}"
            ),
        )
        st.pyplot(fig, width=1300)
