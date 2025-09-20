"""Data loader and config validation for NeuroVoxel app."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import jsonschema
import pandas as pd
from bids.layout import BIDSLayout
from formulaic import (
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    from pathlib import Path

    from bids.layout.models import BIDSImageFile


def load_config(config_file: Path) -> dict[str, Any]:
    """Load and validate a NeuroVoxel configuration file."""
    with config_file.open("r") as f:
        config = json.load(f)

    # better to read in from data/template.json instead
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "NeuroVoxel",
        "description": "NeuroVoxel configuration file schema",
        "type": "object",
        "properties": {
            "paths": {
                "type": "object",
                "properties": {
                    "bids_root": {"type": "string"},
                    "bids_config": {"type": "string"},
                    "tabular": {"type": "string"},
                    "template": {"type": "string"},
                    "mask": {"type": "string"},
                    "outputdir": {"type": "string"},
                },
            },
            "analysis": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "smoothing_fwhm": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 15.0,
                    },
                    "voxel_size": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10.0,
                    },
                    "n_perm": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100000,
                    },
                    "random_seed": {"type": "integer"},
                    "tfce": {"type": "boolean"},
                },
            },
        },
    }

    jsonschema.validate(config, schema)

    return config


def load_bids(
    bids_root: Path,
    config_fname: Path | None = None,
    database_path: Path | None = None,
) -> BIDSLayout:
    """Load BIDS dataset."""
    layout = BIDSLayout(
        bids_root,
        validate=False,
        derivatives=False,
        config=["bids", "derivatives", config_fname] if config_fname else None,
        database_path=database_path,
    )

    layout.add_derivatives(  # pyright: ignore[reportUnknownMemberType]
        bids_root / "derivatives",
        config=["bids", "derivatives", config_fname] if config_fname else None,
    )
    return layout


def parse_layout(layout: BIDSLayout) -> pd.DataFrame:
    """Recreate image-types table from a BIDSLayout.

    Args:
        layout: The BIDS layout object.

    Returns:
        DataFrame of image types.
    """
    # list available imaging outcomes
    img_list: list[BIDSImageFile] = layout.get(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        extension="nii.gz"
    ) + layout.get(extension="nii")  # pyright: ignore[reportUnknownMemberType]
    img_type_counts: dict[tuple[tuple[str, object], ...], int] = {}
    entity_df = pd.DataFrame()

    for img in img_list:
        entities = deepcopy(img.entities)
        for k in ("subject", "session"):
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
        ["SpatialReference", "extension", "tracer"], axis=1, errors="ignore"
    )
    entity_df = entity_df.sort_values(
        by=["datatype", "suffix", "desc", "param", "trc"], na_position="last"
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
