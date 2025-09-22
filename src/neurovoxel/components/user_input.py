"""Parse and validate user text inputs."""

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st


def render_bids_input(autoload: bool = False) -> bool:
    """Get BIDS directory path and config file (optional)."""
    # Use a text input for the BIDS root directory path
    bids_root_box = st.empty()
    bids_root = bids_root_box.text_input(
        "BIDS root directory",
        value=st.session_state.paths.get("bids_root"),
        key="bids_root_input",
    )

    # Update session state if the input changes
    if bids_root:
        st.session_state.paths["bids_root"] = bids_root

    valid_bids = False
    if st.session_state.get("paths", {}).get("bids_root"):
        bids_root_path = Path(
            st.session_state.get("paths", {}).get("bids_root")
        )
        if bids_root_path.is_dir():
            valid_bids = True
        else:
            st.error(f"Directory does not exist: {bids_root_path}")
    else:
        st.write("❗️ No BIDS root directory selected.")

    # Use a text input for a custom BIDS config file
    bids_config_box = st.empty()
    bids_config = bids_config_box.text_input(
        "Optional: Custom BIDS configuration",
        value=st.session_state.paths.get("bids_config"),
        key="bids_config_input",
    )

    # Update session state if the input changes
    if bids_config:
        st.session_state.paths["bids_config"] = bids_config

    bids_config_path: Path | None = None
    if st.session_state.get("paths", {}).get("bids_config"):
        bids_config_path = Path(
            st.session_state.get("paths", {}).get("bids_config")
        )
        if not bids_config_path.is_file():
            valid_bids = False
            st.error(f"File does not exist: {bids_config_path}")
    else:
        st.write(
            "No custom BIDS config file selected. Will use default BIDS config."
        )

    if autoload and valid_bids:
        bids_root_box.empty()
        bids_config_box.empty()

    return valid_bids


def render_template_input(
    label: str,
    var_name: Literal["template", "mask"],
    autoload: bool = False,
) -> bool:
    """Get template or mask path."""
    input_box = st.empty()
    inpt = input_box.text_input(
        label,
        value=st.session_state.paths.get(var_name, ""),
        key=f"{var_name}_input",
    )

    if inpt:
        st.session_state.paths[var_name] = inpt

    valid_input = False
    if st.session_state.get("paths", {}).get(var_name):
        input_path = Path(st.session_state.get("paths", {}).get(var_name))
        if input_path.is_file():
            valid_input = True
        else:
            st.error(f"File does not exist: {input_path}")
    else:
        st.write(f"❗️ No brain {var_name} image selected.")

    if autoload and valid_input:
        input_box.empty()

    return valid_input


def render_table_input(autoload: bool = False) -> bool:
    """Tabular data file."""
    # Use a text input for tabular data file
    tabular_box = st.empty()
    tabular = tabular_box.text_input(
        "Enter tabular data path",
        value=st.session_state.get("paths", {}).get("tabular"),
        key="tabular_input",
    )

    # Update session state if the input changes
    if tabular:
        st.session_state.paths["tabular"] = tabular

    valid_tabular = False
    if st.session_state.get("paths", {}).get("tabular"):
        tabular_path = Path(st.session_state.get("paths", {}).get("tabular"))
        if tabular_path.is_file():
            if tabular_path.suffix in [".csv", ".tsv"]:
                sep = "\t" if tabular_path.suffix == ".tsv" else ","
                # Read only the header first to check columns
                header_df = pd.read_csv(tabular_path, sep=sep, nrows=0)  # pyright: ignore[reportUnknownMemberType]
                required_cols = {"subject", "session"}
                if not required_cols.issubset(header_df.columns):
                    st.error(
                        "Tabular data file must contain "
                        "'subject' and 'session' columns."
                    )
                else:
                    dtype_dict = dict.fromkeys(required_cols, str)
                    st.session_state.tbl = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
                        tabular_path,
                        sep=sep,
                        dtype=dtype_dict,  # pyright: ignore[reportArgumentType]
                    )
                    valid_tabular = True
        else:
            st.error(f"File does not exist: {tabular_path}")
    else:
        st.write("❗️ No tabular file selected.")

    if autoload and valid_tabular:
        tabular_box.empty()

    return valid_tabular


def render_analysis_param_input() -> None:
    """Input for analysis parameters."""
    st.session_state.analysis["smoothing_fwhm"] = st.number_input(
        "Smoothing FWHM (mm) for isotropic Gaussian",
        min_value=0.0,
        max_value=15.0,
        value=st.session_state.get("analysis", {}).get("smoothing_fwhm", 5.0),
        step=0.5,
        key="smoothing_fwhm_input",
    )

    st.session_state.analysis["voxel_size"] = st.number_input(
        "Voxel size of statistical analysis space (mm, isotropic)",
        min_value=1.0,
        max_value=10.0,
        value=st.session_state.get("analysis", {}).get("voxel_size", 4.0),
        step=0.5,
        key="voxel_size_input",
    )

    st.session_state.analysis["n_perm"] = st.number_input(
        "Number of permutations",
        min_value=0,
        max_value=100000,
        value=st.session_state.get("analysis", {}).get("n_perm", 10000),
        step=1,
        key="n_perm_input",
    )

    st.session_state.analysis["tfce"] = st.checkbox(
        "Run TFCE",
        value=st.session_state.get("analysis", {}).get("tfce", False),
        key="tfce_input",
    )


def render_outputdir_input(autoload: bool = False) -> bool:
    """Get output directory path."""
    outputdir_box = st.empty()
    outputdir = outputdir_box.text_input(
        "Output directory",
        value=st.session_state.paths.get("outputdir"),
        key="outputdir_input",
    )

    # Update session state if the input changes
    if outputdir:
        st.session_state.paths["outputdir"] = outputdir

    valid_outputdir = False
    if st.session_state.get("paths", {}).get("outputdir"):
        outputdir_path = Path(
            st.session_state.get("paths", {}).get("outputdir")
        )
        if outputdir_path.exists():
            if is_directory_empty(outputdir_path):
                valid_outputdir = True
            else:
                st.error("Please specify an empty directory for output.")
        else:
            try:
                outputdir_path.mkdir(parents=True, exist_ok=False)
                valid_outputdir = True
            except OSError as e:
                st.error(f"Could not create output directory: {e}")
    else:
        st.write("❗️ No output directory selected.")

    if autoload and valid_outputdir:
        outputdir_box.empty()

    return valid_outputdir


def is_directory_empty(directory_path: Path) -> bool:
    """Check if a directory is empty."""
    if not directory_path.is_dir():
        # Handle cases where the path is not a directory or doesn't exist
        msg = f"Path '{directory_path}' is not a directory or does not exist."
        raise ValueError(msg)

    return not any(directory_path.iterdir())
