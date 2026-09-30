"""NeuroVoxel user application."""

# pyright: reportMissingTypeStubs=false

import argparse
from pathlib import Path

import streamlit as st
from atlasreader import create_output  # type: ignore  # noqa: PGH003

from neurovoxel.components.data import render_entity_table
from neurovoxel.components.footer import render_footer
from neurovoxel.components.header import render_header
from neurovoxel.components.model_runner import render_model_runner
from neurovoxel.components.user_input import (
    parameter_output,
    render_analysis_param_input,
    render_bids_input,
    render_inference_choices,
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
from neurovoxel.utils.output import VERSION, commit, packages, timestamp
from neurovoxel.utils.viz import save_all_maps, save_parameters


def _load_dataset() -> None:
    """Load the configured BIDS dataset into session state."""
    info_loading_bids_box = st.empty()
    info_loading_bids_box.info("Loading BIDS dataset...")
    st.session_state.layout = load_bids(
        bids_root=Path(st.session_state.get("paths", {}).get("bids_root")),
        cache_path=Path(st.session_state.get("paths", {}).get("bids_cache"))
        if st.session_state.get("paths", {}).get("bids_cache")
        else None,
    )
    info_loading_bids_box.empty()
    st.toast("BIDS dataset loaded successfully!")
    st.session_state.entity_df = parse_layout(st.session_state.layout)


def _handle_query() -> tuple[str | None, object | None]:
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
            render_inference_choices(rhs)
    return lhs, rhs


def _save_analysis_results(lhs: str, valid_outputdir: bool) -> None:
    """Render and optionally save the completed analysis."""
    render_model_runner(lhs)
    st.subheader("Results")
    summary_stats = st.session_state.tbl.describe()
    st.dataframe(summary_stats, width="content")  # pyright: ignore[reportUnknownMemberType]
    render_visualization(st.session_state.analysis["inference_terms"])

    if valid_outputdir:
        outpath = Path(st.session_state.get("paths", {}).get("outputdir"))
        analysis = st.session_state.analysis.copy()
        analysis.pop("inference_terms", None)
        statistical_maps = save_all_maps(
            outpath,
            st.session_state.result,
            st.session_state.masker,
            lhs,
        )
        atlasreader_output_path = outpath / "atlasreader_output"
        atlasreader_output_path.mkdir(exist_ok=True)
        create_output(
            outpath / statistical_maps[1],
            cluster_extent=st.session_state["cluster size"],
            atlas="blsa_muse",
            outdir=atlasreader_output_path,
        )
        atlasreader_output = [
            file_path.name for file_path in atlasreader_output_path.glob("*")
        ]
        st.session_state.tbl.to_csv(outpath / "tbl.csv", index=False)
        output_parameters = parameter_output(
            st.session_state.paths,
            analysis,
            VERSION,
            timestamp,
            commit,
            statistical_maps,
            "tbl.csv",
            atlasreader_output,
            packages,
        )
        save_parameters(outpath / "parameters.json", output_parameters)


def main(
    config_file: Path | None = None, autoload: bool = False, debug: bool = False
) -> None:
    """Main entry point for the NeuroVoxel Streamlit app.

    This function is callable directly. When run as a script,
    the argparse-based CLI in the ``__main__`` block parses
    command-line flags and calls this function.
    """
    st.set_page_config(page_title="NeuroVoxel", layout="wide")
    render_header()
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

    with st.container():
        # left/right UI setup
        col1, col2 = st.columns(2)
        with col1:
            valid_bids = render_bids_input(autoload)

            # Button to load dataset, enabled only if
            # BIDS root directory is valid
            load_btn = (
                True
                if autoload and valid_bids
                else st.button("Load BIDS dataset", disabled=not valid_bids)
            )

            if load_btn:
                _load_dataset()

            render_table_input(autoload)
            valid_outputdir = render_outputdir_input(autoload)
        with col2:
            render_template_input("Brain template image", "template", autoload)
            valid_mask = render_template_input(
                "Binary brain mask specifying voxels to analyze",
                "mask",
                autoload,
            )
            with st.expander("Advanced"):
                render_analysis_param_input()

    if "entity_df" in st.session_state:
        # Render editable table; save any user edits to session state
        st.session_state.entity_df = render_entity_table(
            st.session_state.entity_df
        )

    lhs, rhs = _handle_query()

    run_btn = st.button("Run analysis", disabled=not (lhs and valid_mask))
    if run_btn and lhs and rhs is not None:
        _save_analysis_results(lhs, valid_outputdir)

    render_footer()
    if debug:
        with st.expander("Session State"):
            st.write(st.session_state)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments using argparse.

    Mirrors the previous click options:
      --config-file <path>
      --autoload (flag)
      --debug (flag)
    """
    parser = argparse.ArgumentParser(prog="neurovoxel")
    parser.add_argument(
        "--config-file",
        type=Path,
        help="NeuroVoxel configuration file.",
    )
    parser.add_argument(
        "--autoload",
        action="store_true",
        help="Autoload paths",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # argparse will have converted --config-file to a Path (or None)
    main(autoload=args.autoload, debug=args.debug)
