"""NeuroVoxel user application."""

from pathlib import Path

import streamlit as st

from neurovoxel.io import load_bids

if __name__ == "__main__":
    st.title("NeuroVoxel")

    # Use a text input for the BIDS root directory path
    bids_root = st.text_input(
        "Enter BIDS root directory path",
        value=st.session_state.get("bids_root", ""),
        key="bids_root_input",
    )

    # Update session state if the input changes
    if bids_root:
        st.session_state.bids_root = bids_root

    valid_bids_root = False
    if st.session_state.get("bids_root"):
        bids_root_path = Path(st.session_state.bids_root)
        if bids_root_path.is_dir():
            st.write(
                f"Selected BIDS root directory: {st.session_state.bids_root}"
            )
            valid_bids_root = True
        else:
            st.error(f"Directory does not exist: {st.session_state.bids_root}")
    else:
        st.write("No BIDS root directory selected.")

    # Use a text input for any custom BIDS config file
    config_fname = st.text_input(
        "Enter custom BIDS config file path",
        value=st.session_state.get("config_fname", ""),
        key="config_fname_input",
    )

    # Update session state if the input changes
    if config_fname:
        st.session_state.config_fname = config_fname

    valid_config_fname = False
    if st.session_state.get("config_fname"):
        config_fname_path = Path(st.session_state.config_fname)
        if config_fname_path.is_file():
            st.write(
                f"Selected BIDS config file: {st.session_state.config_fname}"
            )
            valid_config_fname = True
        else:
            st.error(f"File does not exist: {st.session_state.config_fname}")
    else:
        st.write("No custom BIDS config file selected.")

    # Button to load dataset, enabled only if BIDS root directory is valid
    load_btn = st.button("Load Dataset", disabled=not valid_bids_root)
    if load_btn and valid_bids_root:
        st.info("Loading dataset...")
        layout = load_bids(
            bids_root=Path(st.session_state.bids_root),
            config_fname=Path(st.session_state.config_fname)
            if st.session_state.config_fname and valid_config_fname
            else None,
        )
        st.success("Dataset loaded successfully!")
