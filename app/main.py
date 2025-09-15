import streamlit as st
from app.components.header import render_header
from app.components.footer import render_footer
from app.features.data_loader import data_loader

st.set_page_config(page_title="NeuroVoxel", layout="wide")
render_header()

dataset_path, run_clicked, use_blsa_str, bids_config_path = data_loader()

if run_clicked:
    st.success(f"Loaded dataset: {dataset_path}")
    if use_bsla_str:
        st.info("BSLA data mode enabled.")
    if bids_config_path:
        st.info(f"Using BIDS config file: {bids_config_path}")

    # Access other paths from the new loader
    paths = st.session_state.get("paths", {})
    if paths:
        st.write("Detected/selected paths:")
        for key, val in paths.items():
            if val:
                st.write(f"- **{key}**: {val}")
            else:
                st.warning(f"- **{key}**: not set")

    # Example: use the edited entity table
    entity_df = st.session_state.get("entity_df")
    if entity_df is not None:
        st.dataframe(entity_df)

render_footer()
