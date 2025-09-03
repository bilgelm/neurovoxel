"""NeuroVoxel user application."""

from copy import deepcopy
from pathlib import Path

import formulaic  # pyright: ignore[reportMissingTypeStubs]
import pandas as pd
import streamlit as st
from bids.layout.models import (  # pyright: ignore[reportMissingTypeStubs]
    BIDSImageFile,
)
from formulaic.errors import (  # pyright: ignore[reportMissingTypeStubs]
    FormulaicError,
    FormulaMaterializationError,
    FormulaSyntaxError,
)

from neurovoxel.analysis import run_query
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
        st.write("❗️ No BIDS root directory selected.")

    # Use a text input for any custom BIDS config file
    config_fname = st.text_input(
        "Optional: Enter custom BIDS config file path",
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
        st.write(
            "No custom BIDS config file selected. Will use default config."
        )

    # Button to load dataset, enabled only if BIDS root directory is valid
    load_btn = st.button("Load Dataset", disabled=not valid_bids_root)
    layout_loaded = False
    if load_btn and valid_bids_root:
        st.info("Loading dataset...")
        layout = load_bids(
            bids_root=Path(st.session_state.bids_root),
            config_fname=Path(st.session_state.config_fname)
            if st.session_state.config_fname and valid_config_fname
            else None,
        )
        st.session_state.layout = layout
        st.success("Dataset loaded successfully!")
        # list available imaging outcomes
        img_list: list[BIDSImageFile] = layout.get(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            extension="nii.gz"
        ) + layout.get(extension="nii")  # pyright: ignore[reportUnknownMemberType]
        img_type_list = {}

        entity_df = pd.DataFrame()
        for img in img_list:
            entities = deepcopy(img.entities)
            del entities["subject"]
            del entities["session"]
            entities_tuple = tuple(entities.items())
            if entities_tuple in img_type_list:
                img_type_list[entities_tuple] += 1
            else:
                entity_df = pd.concat(
                    [entity_df, pd.DataFrame([entities])], ignore_index=True
                )
                img_type_list[entities_tuple] = 1

        entity_df = entity_df.drop(
            ["SpatialReference", "extension", "tracer"], axis=1, errors="ignore"
        )
        entity_df = entity_df.sort_values(
            by=["datatype", "suffix", "desc", "param", "trc"],
        ).reset_index(drop=True)

        def concat_name(row: pd.Series) -> str:
            """Concatenate columns if they exist and are not null.

            Return:
            ------
                Concatenated columns or 'Enter name here' if none are present.
            """
            parts = [
                str(row[col])
                for col in ["desc", "param", "trc", "meas", "suffix"]
                if col in row and pd.notna(row[col])
            ]
            return "_".join(parts) if parts else "Enter name here"

        entity_df["name"] = entity_df.apply(
            concat_name,
            axis=1,
        )
        st.session_state.entity_df = entity_df

    if "entity_df" in st.session_state:
        st.write("Types of images in dataset:")
        # Make only the 'name' column editable
        disabled_cols = [
            col for col in st.session_state.entity_df.columns if col != "name"
        ]
        columns = list(st.session_state.entity_df.columns)
        # move the 'name' column to the beginning
        columns.remove("name")
        columns = ["name", *columns]

        edited_entity_df = st.data_editor(
            st.session_state.entity_df,
            disabled=disabled_cols,
            key="entity_table_editor",
            use_container_width=True,
            column_order=columns,
            hide_index=True,
        )
        # Save edits to session state
        st.session_state.entity_df = edited_entity_df

        # Check for duplicate names
        name_col = edited_entity_df["name"]
        if name_col.duplicated().any():
            st.error(
                "Entries in the 'name' column must be unique. "
                "Please fix duplicates before continuing."
            )

    # Use a text input for tabular data file
    table_fname = st.text_input(
        "Enter tabular data path",
        value=st.session_state.get("table_fname", ""),
        key="table_fname_input",
    )

    # Update session state if the input changes
    if table_fname:
        st.session_state.table_fname = table_fname

    if st.session_state.get("table_fname"):
        table_fname_path = Path(st.session_state.table_fname)
        if table_fname_path.is_file():
            st.write(
                f"Selected tabular data file: {st.session_state.table_fname}"
            )
            valid_table_fname = True
        else:
            st.error(f"File does not exist: {st.session_state.table_fname}")

    # Show query input only if table can be loaded in
    valid_table_fname = False
    valid_query = False
    tbl = None
    lhs = None
    if valid_table_fname:
        table_fname_path = Path(st.session_state.table_fname)
        if table_fname_path.suffix in [".csv", ".tsv"]:
            sep = "\t" if table_fname_path.suffix == ".tsv" else ","
            tbl = pd.read_csv(table_fname_path, sep=sep)  # pyright: ignore[reportUnknownMemberType]
            st.session_state.tbl = tbl

    if "tbl" in st.session_state:
        query = st.text_input(
            "Enter query",
            value=st.session_state.get("query", ""),
            key="query_input",
        )
        if query:
            st.session_state.query = query

            # check query
            # currently we assume imaging is the outcome
            # this check should be rewritten for a more general
            # case that also allows imaging to be on the rhs
            if "~" in query:
                lhs, rhs = query.split("~")
                lhs = lhs.strip()
                rhs = rhs.strip()
                if lhs not in st.session_state.entity_df["name"].values:  # noqa: PD011
                    st.error("Imaging outcome in query is invalid")
                try:
                    X = formulaic.model_matrix(rhs, st.session_state.tbl)  # pyright: ignore[reportUnknownMemberType]
                    valid_query = True
                except FormulaSyntaxError:
                    st.error("Invalid formula syntax")
                except FormulaMaterializationError:
                    st.error(
                        "Issue with materializing the model matrix"
                        "(missing variable or incompatible data)"
                    )
                except (
                    FormulaicError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as e:
                    st.error(f"Error while parsing formula: {e}")
            else:
                st.error("Invalid formula syntax")
    else:
        st.error("Cannot read file: {st.session_state.table_fname}")

    mask = st.text_input(
        "Enter file path to binary mask specifying voxels to analyze",
        value=st.session_state.get("mask", ""),
        key="mask_input",
    )

    # Update session state if the input changes
    if mask:
        st.session_state.mask = mask

    run_btn = st.button("Run analysis", disabled=not valid_query)
    if run_btn and valid_query and lhs:
        st.info("Running analysis...")

        row = st.session_state.entity_df.loc[
            st.session_state.entity_df["name"] == lhs
        ]
        image_paths: list[str] = st.session_state.layout.get(  # pyright: ignore[reportUnknownMemberType]
            **row.to_dict(),  # pyright: ignore[reportCallIssue]
            return_type="filename",
        )

        # call relevant function from neurovoxel
        st.session_state.result = run_query(
            st.session_state.query,
            st.session_state.tbl,
            image_paths,
            st.session_state.mask,
            st.session_state.smoothing_fwhm,
            st.session_state.vox_size,
            st.session_state.n_perm,
            st.session_state.n_jobs,
            st.session_state.random_state,
            st.session_state.tfce,
        )
