"""Footer component for the NeuroVoxel app.

This module provides a function to render the site footer using Streamlit.
"""

import streamlit as st


def render_footer() -> None:
    """Render the site footer for NeuroVoxel app."""
    st.markdown(
        """
        <hr>
        <div style='text-align: center;'>
            <small>2025 NeuroVoxel</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
