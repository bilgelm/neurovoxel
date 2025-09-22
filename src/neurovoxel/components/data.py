"""Display data table."""

import streamlit as st
from pandas import DataFrame


def render_entity_table(entity_df: DataFrame) -> DataFrame:
    """Editable table: only 'name' is editable; validates uniqueness."""
    st.write("Types of images in dataset:")

    # Make only the 'name' column editable
    disabled_cols = [c for c in entity_df.columns if c != "name"]
    cols = list(entity_df.columns)
    # move the 'name' column to the beginning
    if "name" in cols:
        cols.remove("name")
        cols = ["name", *cols]

    edited_df = st.data_editor(
        entity_df,
        disabled=disabled_cols,
        key="entity_table_editor",
        use_container_width=True,
        column_order=cols,
        hide_index=True,
    )

    if "name" in edited_df and edited_df["name"].duplicated().any():
        st.error(
            "Entries in the 'name' column must be unique. "
            "Please fix duplicates before continuing."
        )

    return edited_df
