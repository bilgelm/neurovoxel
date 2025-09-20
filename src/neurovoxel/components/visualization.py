"""Visualization feature for the NeuroVoxel app.

This module provides the UI components for visualizing results using Streamlit.
"""

import matplotlib.pyplot as plt
import streamlit as st
from pandas import Index

from neurovoxel.utils.viz import (
    basic_interactive_viz,  # pyright: ignore[reportUnknownVariableType]
    basic_viz,  # pyright: ignore[reportUnknownVariableType]
)


def render_visualization(rhs: Index) -> None:
    """Render the visualization UI for NeuroVoxel app."""
    for idx, variable in enumerate(rhs):  # pyright: ignore[reportUnknownVariableType, reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
        fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
        basic_viz(
            st.session_state.result,
            st.session_state.masker,
            idx=idx,
            stat="t",
            figure=fig,  # pyright: ignore[reportArgumentType]
            draw_cross=False,  # pyright: ignore[reportArgumentType]
            bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
            transparency=0.5,  # pyright: ignore[reportArgumentType]
            title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
        )
        st.pyplot(fig)

        html_view = basic_interactive_viz(
            st.session_state.result,
            st.session_state.masker,
            idx=idx,
            stat="t",
            bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
            opacity=0.5,  # pyright: ignore[reportArgumentType]
            title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
        )
        html_content = html_view.get_iframe()  # pyright: ignore[reportUnknownMemberType]
        st.components.v1.html(html_content, width=650, height=250)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
