"""Main entry point for the NeuroVoxel Streamlit app.

This module sets up the page, loads data, displays environment banners,
and renders header and footer components.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from app.components.footer import render_footer
from app.components.header import render_header
from app.features.data_loader import data_loader
from app.utils.env_utils import is_testing_env

st.set_page_config(page_title="NeuroVoxel", layout="wide")

render_header()

# Banner for testing environment
config_path = Path.cwd() / "config.yaml"
if is_testing_env(config_path):
    st.warning("**Site is running in TESTING environment**", icon="⚠️")

dataset_path, run_clicked, use_blsa_str, bids_config_path = data_loader()

if run_clicked:
    st.success(f"Loaded dataset: {dataset_path}")
    if use_blsa_str:
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

    entity_df = st.session_state.get("entity_df")
    if entity_df is not None:
        if not isinstance(entity_df, pd.DataFrame):
            entity_df = pd.DataFrame(entity_df)
        st.dataframe(pd.DataFrame(entity_df))

render_footer()
