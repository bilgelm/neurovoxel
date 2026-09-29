"""Data loader and config validation for NeuroVoxel app."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import pandas as pd
from bids2table.pybids import (
    BIDSFile,
    BIDSLayout,  # pyright: ignore[reportUnknownVariableType]
)
from formulaic import (
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)

from neurovoxel.utils import SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def load_config(config_file: Path) -> dict[str, Any]:
    """Load and validate a NeuroVoxel configuration file."""
    with config_file.open("r") as f:
        config = json.load(f)

    jsonschema.validate(config, SCHEMA)

    return config


def load_bids(
    bids_root: Path,
    cache_path: Path | None = None,
) -> BIDSLayout:
    """Load BIDS dataset."""
    return BIDSLayout(
        root=bids_root,  # bids_root/derivatives
        cache_path=cache_path,  # should be corresponding to derivatives
    )


def parse_layout(layout: BIDSLayout) -> pd.DataFrame:
    """Recreate image-types table from a BIDSLayout.

    Args:
        layout: The BIDS layout object.

    Returns:
        DataFrame of image types.
    """
    # list available imaging outcomes
    img_list: list[BIDSFile] = layout.get(extension=".nii.gz") + layout.get(
        extension=".nii"
    )  # pyright: ignore[reportAssignmentType, reportUnknownMemberType]
    img_type_counts: dict[tuple[tuple[str, object], ...], int] = {}
    entity_df = pd.DataFrame()

    for img in img_list:
        entities = deepcopy(img.get_entities())
        for k in ("sub", "ses"):
            entities.pop(k, None)
        key = tuple(entities.items())
        if key in img_type_counts:
            img_type_counts[key] += 1
        else:
            entity_df = pd.concat(
                [entity_df, pd.DataFrame([entities])], ignore_index=True
            )
            img_type_counts[key] = 1

    entity_df = entity_df.drop(
        ["SpatialReference", "ext", "tracer"], axis=1, errors="ignore"
    )
    sort_cols = [
        col
        for col in ["datatype", "suffix", "desc", "param"]
        if col in entity_df.columns
    ]

    entity_df = entity_df.sort_values(
        by=sort_cols,
        na_position="last",
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

    entity_df["name"] = entity_df.apply(concat_name, axis=1)
    return entity_df


def parse_query(
    query: str,
    allowed_lhs_values: list[str],
    rhs_df: pd.DataFrame,
) -> tuple[str, pd.Index]:
    """Parse query to extract the left and right hand side."""
    if "~" in query:
        lhs, rhs = query.split("~")
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs not in allowed_lhs_values:
            msg = "Imaging outcome in query is invalid"
            raise ValueError(msg)
        x_mat = model_matrix(rhs, rhs_df)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return lhs, x_mat.columns  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    msg = "Invalid formula syntax: formula should contain '~'"
    raise ValueError(msg)
