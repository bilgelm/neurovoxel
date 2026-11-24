"""Visualization feature for the NeuroVoxel app.

This module provides the UI components for visualizing results using Streamlit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from pandas import Index

from neurovoxel.utils.viz import (
    basic_interactive_viz,
    basic_viz,
    nanslice_overlay,
)


def render_visualization(
    inference_terms: Index,
    pdf_path: Path | None = None,
) -> None:
    """Render the visualization UI and optionally save a PDF report."""
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    try:
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

            if pdf is not None:
                if pdf_path is None:
                    msg = "PDF output requires a destination path."
                    raise ValueError(msg)
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                static_view = basic_viz(
                    st.session_state.result,
                    st.session_state.masker,
                    term=variable,
                    stat="t",
                    bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
                    display_mode="ortho",  # pyright: ignore[reportArgumentType]
                    title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
                )
                pdf.savefig(  # pyright: ignore[reportUnknownMemberType]
                    static_view.frame_axes.figure,  # pyright: ignore[reportUnknownMemberType]
                    bbox_inches="tight",
                )
                static_view.close()

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

            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(fig, Figure):  # pyright: ignore[reportUnnecessaryIsInstance]
                plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
