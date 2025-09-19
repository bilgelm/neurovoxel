"""NeuroVoxel user application."""

# pyright: reportMissingTypeStubs=false

from pathlib import Path

import click
import matplotlib.pyplot as plt
import streamlit as st

from neurovoxel.components.data import render_entity_table
from neurovoxel.components.footer import render_footer
from neurovoxel.components.header import render_header
from neurovoxel.components.text_input import (
    render_bids_input,
    render_table_input,
    render_template_input,
)
from neurovoxel.utils.analysis import get_masker, run_query
from neurovoxel.utils.load_parse import (
    load_bids,
    load_config,
    parse_layout,
    parse_query,
)
from neurovoxel.utils.viz import (
    basic_interactive_viz,  # pyright: ignore[reportUnknownVariableType]
    basic_viz,  # pyright: ignore[reportUnknownVariableType]
)


@click.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
    help="NeuroVoxel configuration file.",
)
@click.option("--autoload", is_flag=True, help="Autoload paths")
@click.option("--test-mode", is_flag=True, help="Enable test mode")
def main(
    config_file: Path | None = None,
    autoload: bool = False,
    test_mode: bool = False,
) -> None:
    """Main entry point for the NeuroVoxel Streamlit app."""
    st.set_page_config(page_title="NeuroVoxel", layout="wide")
    print(f"autoload {autoload}")

    render_header()

    # Banner for testing environment
    if test_mode:
        st.warning("**Site is running in TESTING environment**", icon="⚠️")

    st.session_state.setdefault("paths", {})
    st.session_state.setdefault("analysis", {})
    if config_file:
        st.info(f"Using NeuroVoxel configuration file: {config_file}")
        config = load_config(Path(config_file))
        st.success(
            "Configuration file loaded and validated! "
            "Inputs will be pre-filled with values from the configuration file."
        )
        if "paths" in config:
            st.session_state.paths.update(config["paths"])
        if "analysis" in config:
            st.session_state.analysis.update(config["analysis"])

    col1, col2 = st.columns(2)
    with col1:
        valid_bids = render_bids_input(autoload)

        # Button to load dataset, enabled only if BIDS root directory is valid
        load_btn = (
            True
            if autoload and valid_bids
            else st.button("Load BIDS dataset", disabled=not valid_bids)
        )

        if load_btn:
            info_loading_bids_box = st.empty()
            info_loading_bids_box.info("Loading BIDS dataset...")
            layout = load_bids(
                bids_root=Path(
                    st.session_state.get("paths", {}).get("bids_root")
                ),
                config_fname=Path(
                    st.session_state.get("paths", {}).get("bids_config")
                ),
            )
            st.session_state.layout = layout
            info_loading_bids_box.empty()
            st.success("BIDS dataset loaded successfully!")
            st.session_state.entity_df = parse_layout(layout)

        valid_table = render_table_input(autoload)
    with col2:
        valid_tpl = render_template_input(
            "Brain template image", "template", autoload
        )
        valid_mask = render_template_input(
            "Binary brain mask specifying voxels to analyze", "mask", autoload
        )
        with st.expander("Advanced"):
            smoothing_fwhm = st.number_input(
                "Smoothing FWHM (mm) for isotropic Gaussian",
                min_value=0.0,
                max_value=15.0,
                value=st.session_state.get("analysis", {}).get(
                    "smoothing_fwhm", 5.0
                ),
                step=0.5,
                key="smoothing_fwhm_input",
            )
            if smoothing_fwhm:
                st.session_state.analysis["smoothing_fwhm"] = smoothing_fwhm

            voxel_size = st.number_input(
                "Voxel size of statistical analysis space (mm, isotropic)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.get("analysis", {}).get(
                    "voxel_size", 4.0
                ),
                step=0.5,
                key="voxel_size_input",
            )
            if voxel_size:
                st.session_state.analysis["voxel_size"] = voxel_size

            n_perm = st.number_input(
                "Number of permutations",
                min_value=0,
                max_value=100000,
                value=st.session_state.get("analysis", {}).get("n_perm", 1000),
                step=1,
                key="n_perm_input",
            )
            if n_perm:
                st.session_state.analysis["n_perm"] = n_perm

            tfce = st.checkbox(
                "Run TFCE",
                value=st.session_state.get("analysis", {}).get("tfce", False),
                key="tfce_input",
            )

            if tfce:
                st.session_state.analysis["tfce"] = tfce

    if "entity_df" in st.session_state:
        # Render editable table; save any user edits to session state
        st.session_state.entity_df = render_entity_table(
            st.session_state.entity_df
        )

    query = st.text_input(
        "Enter query",
        value=st.session_state.get("analysis", {}).get("query"),
        key="query_input",
    )

    if query:
        st.session_state.analysis["query"] = query

    lhs = rhs = None
    if query and "entity_df" in st.session_state:
        lhs, rhs = parse_query(
            query,
            st.session_state.entity_df["name"].tolist(),
            st.session_state.tbl,
        )

    print(f"lhs {lhs}")
    print(f"rhs {rhs}")

    run_btn = st.button(
        "Run analysis",
        disabled=not (lhs and valid_mask),
    )
    if run_btn:
        st.info("Running analysis...")

        row = (
            st.session_state.entity_df.loc[
                st.session_state.entity_df["name"] == lhs
            ]
            .drop(columns=["name"])
            .dropna(axis=1)  # pyright: ignore[reportUnknownMemberType]
        )
        images = st.session_state.layout.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            **row.to_dict(orient="records")[0],  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
        )

        st.info(f"Analysis will include up to {len(images)} {lhs} images")

        # call relevant function from neurovoxel
        masker = get_masker(
            st.session_state.get("paths", {}).get("mask"),
            st.session_state.get("analysis", {}).get("smoothing_fwhm"),
            st.session_state.get("analysis", {}).get("voxel_size"),
            n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
        )

        st.session_state.result = run_query(
            st.session_state.get("analysis", {}).get("query"),
            st.session_state.tbl,
            images,  # pyright: ignore[reportUnknownArgumentType]
            masker,
            n_perm=st.session_state.get("analysis", {}).get("n_perm"),
            n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
            random_state=st.session_state.get("analysis", {}).get(
                "random_state", 42
            ),
            tfce=tfce,
        )

        for idx, variable in enumerate(rhs):  # pyright: ignore[reportUnknownVariableType, reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
            fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
            basic_viz(
                st.session_state.result,
                masker,
                idx=idx,
                stat="t",
                figure=fig,  # pyright: ignore[reportArgumentType]
                draw_cross=False,  # pyright: ignore[reportArgumentType]
                bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
                transparency=0.5,  # pyright: ignore[reportArgumentType]
                title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
            )
            st.pyplot(fig)

            html_view = basic_interactive_viz(
                st.session_state.result,
                masker,
                idx=idx,
                stat="t",
                bg_img=st.session_state.get("paths", {}).get("template"),  # pyright: ignore[reportArgumentType]
                opacity=0.5,  # pyright: ignore[reportArgumentType]
                title=f"t-stat for {variable}",  # pyright: ignore[reportArgumentType]
            )
            html_content = html_view.get_iframe()  # pyright: ignore[reportUnknownMemberType]
            st.components.v1.html(html_content, width=650, height=250)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    render_footer()


if __name__ == "__main__":
    main()
