"""UI components for the NeuroVoxel model runner feature."""

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import streamlit as st

from neurovoxel.utils import STANDARDIZATION_OPTS
from neurovoxel.utils.analysis import get_masker, run_query
from neurovoxel.utils.formula import split_formula

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

PROGRESS_UPDATE_INTERVAL = 0.1


@contextmanager
def _lme_progress(
    *, prepare: bool = False
) -> Iterator[Callable[[int, int], None]]:
    """Render throttled progress on the UI thread and clear it on exit."""
    bar: DeltaGenerator | None = None
    last_update = 0.0

    def update(completed: int, total: int) -> None:
        nonlocal bar, last_update
        now = monotonic()
        if (
            bar is not None
            and 0 < completed < total
            and now - last_update < PROGRESS_UPDATE_INTERVAL
        ):
            return
        fraction = completed / total
        label = f"Fitting LME: {completed:,}/{total:,} voxels ({fraction:.0%})"
        if bar is None:
            bar = st.progress(fraction, text=label)
        else:
            bar.progress(fraction, text=label)
        last_update = now

    try:
        if prepare:
            bar = st.progress(
                0.0,
                text="Preparing LME data: matching images, loading and "
                "preprocessing...",
            )
        yield update
    finally:
        if bar is not None:
            bar.empty()


def render_model_runner(lhs: str) -> None:
    """Render the model runner UI for NeuroVoxel app."""
    query = st.session_state.analysis.get("query", "")
    mixed = split_formula(query).random is not None
    with _lme_progress(prepare=mixed) as progress:
        _run_model(lhs, progress)


def _run_model(lhs: str, progress: Callable[[int, int], None]) -> None:
    """Load the selected images and run the analysis with an active UI."""
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

    with st.spinner("Running analysis..."):
        # call relevant function from neurovoxel
        st.session_state.masker = get_masker(
            st.session_state.get("paths", {}).get("mask"),
            st.session_state.get("analysis", {}).get("smoothing_fwhm"),
            st.session_state.get("analysis", {}).get("voxel_size"),
            n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            st.session_state.result, st.session_state.tbl = run_query(
                st.session_state.get("analysis", {}).get("query"),
                Path(st.session_state.get("paths", {}).get("bids_root")),
                st.session_state.get("analysis", {}).get("inference_terms"),
                st.session_state.tbl,
                images,  # pyright: ignore[reportUnknownArgumentType]
                st.session_state.masker,
                correction=st.session_state.analysis.get("correction", "fwe"),
                progress_callback=progress,
                voxelwise_standardization=st.session_state.analysis.get(
                    "voxelwise_standardization", STANDARDIZATION_OPTS[0]
                ),
                n_perm=st.session_state.get("analysis", {}).get("n_perm"),
                n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
                random_state=st.session_state.get("analysis", {}).get(
                    "random_state", 42
                ),
                tfce=st.session_state.get("analysis", {}).get("tfce"),
                handle_zero_voxels=st.session_state.get("analysis", {}).get(
                    "handle_zero_voxels"
                ),
                handle_multiple_sessions=st.session_state.get(
                    "analysis", {}
                ).get("handle_multiple_sessions"),
            )
        for warning in caught:
            st.warning(str(warning.message))
