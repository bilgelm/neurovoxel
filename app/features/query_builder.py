"""Streamlit UI for building and submitting neuroimaging analysis queries.

This module provides a query builder form with variable buttons, advanced
settings, and analysis submission functionality.
"""

import streamlit as st


def query_builder(image_names: list[str]) -> dict[str, str | bool | float | int]:
    """Display a query builder form with variable buttons.

    Clicking a variable button appends it to the query field.
    """
    st.subheader("Query Builder")

    # Use session state to persist query string
    if "query_string" not in st.session_state:
        st.session_state["query_string"] = ""

    # Show variable buttons
    st.write("Click a variable to add it to your query:")
    cols = st.columns(min(len(image_names), 5))
    for idx, name in enumerate(image_names):
        if cols[idx % 5].button(name, key=f"varbtn_{name}"):
            # Only add variable if not already present as a distinct token
            import re

            current_query = st.session_state["query_string"]
            # Tokenize by space, +, ~, =, etc.
            tokens = re.split(r"[\s\+~=]+", current_query)
            if name not in tokens:
                if current_query and not current_query.endswith(
                    (" ", "+", "~", "=")
                ):
                    st.session_state["query_string"] += f" + {name}"
                else:
                    st.session_state["query_string"] += name

    # Query field and Run button side by side
    # Query field
    query_string = st.text_input(
        "Build your query by clicking variables and/or typing:",
        value=st.session_state["query_string"],
        key="query_string_input",
    )

    # Prefill advanced settings from config.yaml if present
    config_analysis = st.session_state.get("config_analysis", {})
    config_source = st.session_state.get("config_source", "default")
    with st.expander("Advanced"):
        smoothing_default = float(config_analysis.get("smoothing_fwhm", 5.0))
        smoothing = st.number_input(
            "Smoothing FWHM (mm) for isotropic Gaussian",
            min_value=0.0,
            value=smoothing_default,
            step=0.5,
            format="%.2f",
            key="smoothing_fwhm",
        )
        if config_source == "uploaded" and "smoothing_fwhm" in config_analysis:
            st.caption("Prepopulated from uploaded config.yaml")

        voxel_size_default = float(config_analysis.get("voxel_size", 6.0))
        voxel_size = st.number_input(
            "Voxel size of statistical analysis space (mm, isotropic)",
            min_value=0.0,
            value=voxel_size_default,
            step=0.5,
            format="%.2f",
            key="voxel_size",
        )
        if config_source == "uploaded" and "voxel_size" in config_analysis:
            st.caption("Prepopulated from uploaded config.yaml")

        permutations_default = int(config_analysis.get("permutations", 100))
        permutations = st.number_input(
            "Number of permutations",
            min_value=1,
            value=permutations_default,
            step=1,
            key="num_permutations",
        )
        if config_source == "uploaded" and "permutations" in config_analysis:
            st.caption("Prepopulated from uploaded config.yaml")

        run_tfce = st.checkbox("Run TFCE", key="run_tfce")

    # Run analysis button after advanced settings
    run_clicked = st.button("Run analysis", key="run_analysis_btn")

    # Sync session state with manual edits
    st.session_state["query_string"] = query_string

    if run_clicked:
        st.success(f"Analysis submitted: {query_string}")

    st.caption(
        "Click variable buttons to add them to your query, or type manually. Then click 'Run analysis'."
    )
    return {
        "query": query_string if query_string is not None else "",
        "run_clicked": run_clicked,
        "smoothing_fwhm": smoothing,
        "voxel_size": voxel_size,
        "num_permutations": permutations,
        "run_tfce": run_tfce,
    }
