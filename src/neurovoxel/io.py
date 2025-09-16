"""Data I/O."""

# pyright: reportMissingTypeStubs = false

from pathlib import Path
from re import sub
from typing import Any, Literal, get_args

import numpy as np
from bids.layout import BIDSLayout
from nibabel.loadsave import (
    save as nib_save,  # pyright: ignore[reportUnknownVariableType]
)
from nilearn.maskers import MultiNiftiMasker

from neurovoxel.viz import unmask

STAT_OPTIONS_LITERAL = Literal[
    "t",
    "logp_max_t",
    "tfce",
    "logp_max_tfce",
    "size",
    "logp_max_size",
    "mass",
    "logp_max_mass",
]
STAT_OPTIONS = get_args(STAT_OPTIONS_LITERAL)


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


def save_stat_map(
    filename: Path,
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: STAT_OPTIONS_LITERAL = "t",
) -> None:
    """Save stat map."""
    nib_save(unmask(result[stat][idx, :], masker), filename)


def save_all_maps(
    outputdir: Path,
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    imgvar: str,
    testedvars: list[str],
) -> None:
    """Save all stat maps."""
    outputdir.mkdir(parents=True, exist_ok=True)

    for stat in STAT_OPTIONS:
        if stat in result:
            for idx in range(result[stat].shape[0]):
                # <source>_contrast-<label>_stat-<label>_<mod>map.nii.gz
                testedvar = sub(r"[^\w]", "", testedvars[idx])
                filename = (
                    outputdir
                    / f"{imgvar}_contrast-{testedvar}_stat-{stat}_map.nii.gz"
                )
                save_stat_map(filename, result, masker, idx, stat)
