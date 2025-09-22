"""Header component for the NeuroVoxel app.

This module provides the render_header function to display
the app's logo and tagline.
"""

import streamlit as st


def render_header() -> None:
    """Render the header for NeuroVoxel app."""
    st.markdown(
        (
            "<div style='display:flex; flex-direction:column; "
            "align-items:center;'>"
        ),
        unsafe_allow_html=True,
    )
    st.image("assets/logo.png", width=500)
    st.markdown(
        (
            "<div style='color:#AAA; font-size:1.1rem; margin-top:8px;'>"
            "Statistical Analysis Platform for Neuroimaging Datasets"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
