"""UI components for the NeuroVoxel model runner feature."""

import streamlit as st

from neurovoxel.utils.analysis import get_masker, run_query


def render_model_runner(lhs: str) -> None:
    """Render the model runner UI for NeuroVoxel app."""
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

    with st.spinner("Running analysis..."):
        # call relevant function from neurovoxel
        st.session_state.masker = get_masker(
            st.session_state.get("paths", {}).get("mask"),
            st.session_state.get("analysis", {}).get("smoothing_fwhm"),
            st.session_state.get("analysis", {}).get("voxel_size"),
            n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
        )

        st.session_state.result = run_query(
            st.session_state.get("analysis", {}).get("query"),
            st.session_state.tbl,
            images,  # pyright: ignore[reportUnknownArgumentType]
            st.session_state.masker,
            n_perm=st.session_state.get("analysis", {}).get("n_perm"),
            n_jobs=st.session_state.get("analysis", {}).get("n_jobs", -1),
            random_state=st.session_state.get("analysis", {}).get(
                "random_state", 42
            ),
            tfce=st.session_state.get("analysis", {}).get("tfce"),
        )
