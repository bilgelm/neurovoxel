"""NeuroVoxel user application."""

# pyright: reportMissingTypeStubs=false

from pathlib import Path

import click
import streamlit as st

from neurovoxel.components.data import render_entity_table
from neurovoxel.components.footer import render_footer
from neurovoxel.components.header import render_header
from neurovoxel.components.model_runner import render_model_runner
from neurovoxel.components.user_input import (
    render_analysis_param_input,
    render_bids_input,
    render_outputdir_input,
    render_table_input,
    render_template_input,
)
from neurovoxel.components.visualization import render_visualization
from neurovoxel.utils.load_parse import (
    load_bids,
    load_config,
    parse_layout,
    parse_query,
)
from neurovoxel.utils.viz import save_all_maps


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

    render_header()

    # Banner for testing environment
    if test_mode:
        st.warning("**Site is running in TESTING environment**", icon="⚠️")

    st.session_state.setdefault("paths", {})
    st.session_state.setdefault("analysis", {})
    if config_file:
        st.info(f"Using NeuroVoxel configuration file: {config_file}")
        config = load_config(Path(config_file))
        st.toast(
            "Configuration file loaded and validated! "
            "Inputs will be pre-filled."
        )
        st.session_state.paths.update(config.get("paths", {}))
        st.session_state.analysis.update(config.get("analysis", {}))

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
            st.session_state.layout = load_bids(
                bids_root=Path(
                    st.session_state.get("paths", {}).get("bids_root")
                ),
                config_fname=Path(
                    st.session_state.get("paths", {}).get("bids_config")
                ),
            )
            info_loading_bids_box.empty()
            st.toast("BIDS dataset loaded successfully!")
            st.session_state.entity_df = parse_layout(st.session_state.layout)

        render_table_input(autoload)
        valid_outputdir = render_outputdir_input(autoload)
    with col2:
        render_template_input("Brain template image", "template", autoload)
        valid_mask = render_template_input(
            "Binary brain mask specifying voxels to analyze", "mask", autoload
        )
        with st.expander("Advanced"):
            render_analysis_param_input()

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

    lhs = rhs = None
    if query:
        st.session_state.analysis["query"] = query
        if "entity_df" in st.session_state:
            lhs, rhs = parse_query(
                st.session_state.analysis["query"],
                st.session_state.entity_df["name"].tolist(),
                st.session_state.tbl,
            )

    run_btn = st.button(
        "Run analysis",
        disabled=not (lhs and valid_mask),
    )
    if run_btn and lhs and (rhs is not None):
        render_model_runner(lhs)

        st.subheader("Results")
        render_visualization(rhs)

        if valid_outputdir:
            save_all_maps(
                Path(st.session_state.get("paths", {}).get("outputdir")),
                st.session_state.result,
                st.session_state.masker,
                lhs,
                rhs.to_numpy().tolist(),  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownMemberType, reportUnknownArgumentType]
            )

    render_footer()


if __name__ == "__main__":
    main()
