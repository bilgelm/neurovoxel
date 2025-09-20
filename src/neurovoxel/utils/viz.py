"""Visualize maps resulting from statistical analysis."""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownVariableType=false

from pathlib import Path
from re import sub
from typing import Any, Literal, get_args

import numpy as np
from nibabel.loadsave import (
    save as nib_save,  # pyright: ignore[reportUnknownVariableType]
)
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, view_img
from nilearn.plotting.displays._slicers import OrthoSlicer
from nilearn.plotting.html_stat_map import StatMapView

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


def unmask(
    masked_img: np.typing.NDArray[Any], masker: MultiNiftiMasker
) -> Nifti1Image:
    """Convert a rasterized masked image back to an image."""
    return masker.inverse_transform(masked_img)


def basic_viz(
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: STAT_OPTIONS_LITERAL = "t",
    **kwargs: dict[str, Any],
) -> OrthoSlicer:
    """Simple visualization."""
    return plot_stat_map(
        unmask(result[stat][idx, :], masker),
        **kwargs,
    )


def basic_interactive_viz(  # pyright: ignore[]
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: STAT_OPTIONS_LITERAL = "t",
    **kwargs: dict[str, Any],
) -> StatMapView:
    """Simple interactive visualization."""
    return view_img(
        unmask(result[stat][idx, :], masker),
        **kwargs,
    )


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
