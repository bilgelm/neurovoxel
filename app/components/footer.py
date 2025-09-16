import streamlit as st


def render_footer() -> None:
    """Render the site footer."""
    st.markdown(
        """
    <hr>
    <div style='text-align: center;'>
        <small>2025 NeuroVoxel</small>
    </div>
    """,
        unsafe_allow_html=True,
    )
