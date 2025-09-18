"""NeuroVoxel user application."""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import nanslice as ns
from bids.layout.models import (  # pyright: ignore[reportMissingTypeStubs]
    BIDSImageFile,
)
from formulaic import (  # pyright: ignore[reportMissingTypeStubs]
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)
from formulaic.errors import (  # pyright: ignore[reportMissingTypeStubs]
    FormulaicError,
    FormulaMaterializationError,
    FormulaSyntaxError,
)

from neurovoxel.analysis import get_masker, run_query
from neurovoxel.io import load_bids, save_all_maps
from neurovoxel.viz import (
    basic_interactive_viz,  # pyright: ignore[reportUnknownVariableType]
    basic_viz,  # pyright: ignore[reportUnknownVariableType]
    nanslice_overlay,  # pyright: ignore[reportUnknownVariableType]
)

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
    valid_table = False
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
            if table_fname_path.suffix in [".csv", ".tsv"]:
                sep = "\t" if table_fname_path.suffix == ".tsv" else ","
                # Read only the header first to check columns
                header_df = pd.read_csv(table_fname_path, sep=sep, nrows=0)  # pyright: ignore[reportUnknownMemberType]
                required_cols = {"subject", "session"}
                if not required_cols.issubset(header_df.columns):
                    st.error(
                        "Tabular data file must contain "
                        "'subject' and 'session' columns."
                    )
                else:
                    dtype_dict = dict.fromkeys(required_cols, str)
                    st.session_state.tbl = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
                        table_fname_path,
                        sep=sep,
                        dtype=dtype_dict,  # pyright: ignore[reportArgumentType]
                    )
                    valid_table = True
        else:
            st.error(f"File does not exist: {st.session_state.table_fname}")

    # Show query input only if table can be loaded in
    valid_query = False
    lhs = None
    query = st.text_input(
        "Enter query",
        value=st.session_state.get("query", ""),
        key="query_input",
        disabled=not valid_table,
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
                x_mat = model_matrix(rhs, st.session_state.tbl)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
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

    bg_img = st.text_input(
        "Enter file path to brain template image",
        value=st.session_state.get("bg_img", ""),
        key="bg_img_input",
    )

    if bg_img:
        st.session_state.bg_img = bg_img

    valid_bg_img = False
    if st.session_state.get("bg_img"):
        bg_img_path = Path(st.session_state.bg_img)
        if bg_img_path.is_file():
            st.write(
                f"Selected brain template image: {st.session_state.bg_img}"
            )
            valid_bg_img = True
        else:
            st.error(f"File does not exist: {st.session_state.bg_img}")
    else:
        st.write("❗️ No brain template image selected.")

    mask = st.text_input(
        "Enter file path to binary mask specifying voxels to analyze",
        value=st.session_state.get("mask", ""),
        key="mask_input",
    )

    if mask:
        st.session_state.mask = mask

    valid_mask = False
    if st.session_state.get("mask"):
        mask_path = Path(st.session_state.mask)
        if mask_path.is_file():
            st.write(f"Selected brain mask: {st.session_state.mask}")
            valid_mask = True
        else:
            st.error(f"File does not exist: {st.session_state.mask}")
    else:
        st.write("❗️ No brain mask selected.")

    smoothing_fwhm = st.number_input(
        "Smoothing FWHM (mm) for isotropic Gaussian",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        key="smoothing_fwhm_input",
    )

    vox_size = st.number_input(
        "Voxel size of statistical analysis space (mm, isotropic)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        key="vox_size",
    )

    n_perm = st.number_input(
        "Number of permutations",
        min_value=1,
        max_value=100000,
        value=1000,
        step=1,
        key="n_perm",
    )

    tfce = st.checkbox(
        "Run TFCE",
        value=False,
        key="tfce",
    )

    outputdir = st.text_input(
        "Enter directory to save outputs to",
        value=st.session_state.get("outputdir", ""),
        key="outputdir_input",
    )

    if outputdir:
        st.session_state.outputdir = outputdir

    valid_outputdir = False
    if st.session_state.get("outputdir"):
        outputdir_path = Path(st.session_state.outputdir)
        if outputdir_path.is_dir():
            st.write(f"Output directory: {st.session_state.outputdir}")
            valid_outputdir = True
        else:
            st.error(
                f"Output directory does not exist: {st.session_state.outputdir}"
            )
    else:
        st.write("❗️ Output directory is not specified.")

    run_btn = st.button(
        "Run analysis",
        disabled=not valid_query
        or not valid_bg_img
        or not valid_mask
        or not valid_outputdir,
    )
    if run_btn and valid_query and lhs:
        st.info("Running analysis...")

        row = (
            st.session_state.entity_df.loc[
                st.session_state.entity_df["name"] == lhs
            ]
            .drop(columns=["name"])
            .dropna(axis=1)  # pyright: ignore[reportUnknownMemberType]
        )
        images: list[BIDSImageFile] = st.session_state.layout.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            **row.to_dict(orient="records")[0],  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
        )

        st.info(f"Analysis will include up to {len(images)} {lhs} images")  # pyright: ignore[reportUnknownArgumentType]

        # call relevant function from neurovoxel
        masker = get_masker(
            st.session_state.mask,
            smoothing_fwhm,
            vox_size,
            n_jobs=-1,
        )
        st.session_state.result = run_query(
            st.session_state.query,
            st.session_state.tbl,
            images,  # pyright: ignore[reportUnknownArgumentType]
            masker,
            n_perm,
            n_jobs=-1,
            random_state=42,
            tfce=tfce,
        )

        save_all_maps(
            Path(st.session_state.outputdir),
            st.session_state.result,
            masker,
            lhs,
            x_mat.columns.to_numpy().tolist(),  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
        )

        for idx, variable in enumerate(x_mat.columns):  # pyright: ignore[reportUnknownVariableType, reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
            # fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
            # display = basic_viz(
            #     st.session_state.result,
            #     masker,
            #     idx=idx,
            #     stat="t",
            #     figure=fig,  # pyright: ignore[reportArgumentType]
            #     draw_cross=False,  # pyright: ignore[reportArgumentType]
            #     bg_img=st.session_state.bg_img,  # pyright: ignore[reportArgumentType]
            #     transparency=0.5,  # pyright: ignore[reportArgumentType]
            #     title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
            # )
            fig = nanslice_overlay(
                result=st.session_state.result,
                masker=masker,
                idx=idx,
                alpha_stat="t",
                contour_stat="logp_max_t",   # <--- use this
                alpha_lim=(-3, 3),
                cmap="RdYlBu_r",
                title=f"Predictor {idx}: Beta overlay with t-stat transparency + p<0.05 contours"
            )

            st.pyplot(fig)

            html_view = basic_interactive_viz(
                st.session_state.result,
                masker,
                idx=idx,
                stat="t",
                bg_img=st.session_state.bg_img,  # pyright: ignore[reportArgumentType]
                opacity=0.5,  # pyright: ignore[reportArgumentType]
                title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
            )
            html_content = html_view.get_iframe()  # pyright: ignore[reportUnknownMemberType]
            st.components.v1.html(html_content, width=650, height=250)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
