"""Visualize maps resulting from statistical analysis."""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownVariableType=false

from pathlib import Path
from re import sub
from typing import Any, Literal, get_args

import numpy as np
from nanslice import Layer
from nanslice.jupyter import slices
from nibabel.loadsave import (
    save as nib_save,  # pyright: ignore[reportUnknownVariableType]
)
from nibabel.nifti1 import Nifti1Image
from nilearn.image import math_img
from nilearn.maskers import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, view_img
from nilearn.plotting.displays._slicers import OrthoSlicer
from nilearn.plotting.html_stat_map import StatMapView

from neurovoxel.utils.analysis import PermutedOLSResult

# see: https://nilearn.github.io/stable/modules/generated/nilearn.mass_univariate.permuted_ols.html # noqa: E501
STAT_OPTIONS_LITERAL = Literal[
    "beta",  # regression coefficient
    "t",  # t-statistic associated with the significance test
    "tfce",  # TFCE values associated with the significance test
    "size",  # cluster size values associated with the significance test
    "mass",  # cluster mass values associated with the significance test
]
STAT_OPTIONS = get_args(STAT_OPTIONS_LITERAL)

P_OPTIONS_LITERAL = Literal[
    "logp_max_t",  # -log10 family-wise corrected p-values
    "logp_max_tfce",  # -log10 family-wise corrected p-values
    "logp_max_size",  # -log10 family-wise corrected cluster-level p-values
    "logp_max_mass",  # -log10 family-wise corrected cluster-level p-values
]

P_OPTIONS = get_args(P_OPTIONS_LITERAL)


def unmask(
    masked_img: np.typing.NDArray[Any], masker: MultiNiftiMasker
) -> Nifti1Image:
    """Convert a rasterized masked image back to an image."""
    return masker.inverse_transform(masked_img)


def basic_viz(
    result: PermutedOLSResult,
    masker: MultiNiftiMasker,
    term: str,
    stat: Literal[STAT_OPTIONS_LITERAL, P_OPTIONS_LITERAL] = "t",
    **kwargs: dict[str, Any],
) -> OrthoSlicer:
    """Simple visualization."""
    if stat in result:
        idx = result["tested_var_names"].index(term)
        return plot_stat_map(
            unmask(result[stat][idx, :], masker),  # pyright: ignore[reportTypedDictNotRequiredAccess]
            **kwargs,
        )
    msg = f"{stat} is not in results"
    raise KeyError(msg)


def basic_interactive_viz(  # pyright: ignore[]
    result: PermutedOLSResult,
    masker: MultiNiftiMasker,
    term: str,
    stat: Literal[STAT_OPTIONS_LITERAL, P_OPTIONS_LITERAL] = "t",
    **kwargs: dict[str, Any],
) -> StatMapView:
    """Simple interactive visualization."""
    if stat in result:
        idx = result["tested_var_names"].index(term)
        return view_img(
            unmask(result[stat][idx, :], masker),  # pyright: ignore[reportTypedDictNotRequiredAccess]
            **kwargs,
        )
    msg = f"{stat} is not in results"
    raise KeyError(msg)


def nanslice_overlay(  # noqa: PLR0913
    result: PermutedOLSResult,
    masker: MultiNiftiMasker,
    bg_img: Nifti1Image,
    term: str,
    p_var: P_OPTIONS_LITERAL = "logp_max_t",
    p_thresh: float = 0.05,
    cmap: str = "RdYlBu_r",
    title: str = "Beta overlay",
) -> None:
    """Dual-coded visualization.

    Visualize beta coefficients as overlay, alpha controlled by t-statistics,
    contours from -log10 p-values (e.g., logp_max_t).
    """
    if p_var not in result:
        msg = f"Specified p-value variable {p_var} is not in results"
        raise ValueError(msg)

    idx = result["tested_var_names"].index(term)
    beta_img = unmask(result["beta"][idx, :], masker)
    alpha_img = math_img(
        "1 - np.power(10, -img)",
        img=unmask(result[p_var][idx, :], masker),  # pyright: ignore[reportTypedDictNotRequiredAccess]
    )
    contour_level = 1 - p_thresh

    # Create nanslice layers
    base_layer = Layer(
        bg_img,
        cmap="gray",  # gist_gray?
        label="Brain template",
    )

    beta_layer = Layer(
        beta_img,
        cmap=cmap,
        alpha=alpha_img,
        alpha_lim=(0, 1),
        alpha_label="1 - p",
        label="Regression coefficient",
    )

    # Plot with nanslice
    return slices(
        [base_layer, beta_layer],
        nrows=5,
        ncols=5,
        cbar=1,
        contour=contour_level,
        title=title,
    )


def save_stat_map(
    filename: Path,
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: Literal[STAT_OPTIONS_LITERAL, P_OPTIONS_LITERAL] = "t",
) -> None:
    """Save stat map."""
    nib_save(unmask(result[stat][idx, :], masker), filename)


def save_all_maps(
    outputdir: Path,
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    imgvar: str,
) -> None:
    """Save all stat maps."""
    testedvars = result["tested_var_names"]
    outputdir.mkdir(parents=True, exist_ok=True)

    for stat in STAT_OPTIONS + P_OPTIONS:
        if stat in result:
            for idx in range(result[stat].shape[0]):
                # <source>_contrast-<label>_stat-<label>_<mod>map.nii.gz
                testedvar = sub(r"[^\w]", "", testedvars[idx])
                stt = stat.replace("_", "")
                filename = (
                    outputdir
                    / f"{imgvar}_contrast-{testedvar}_stat-{stt}_map.nii.gz"
                )
                save_stat_map(filename, result, masker, idx, stat)
