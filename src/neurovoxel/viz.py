"""Visualize maps resulting from statistical analysis."""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownVariableType=false

from typing import Any

import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, view_img
from nilearn.plotting.displays._slicers import OrthoSlicer
from nilearn.plotting.html_stat_map import StatMapView
import nanslice as ns
from nanslice.jupyter import three_plane

def unmask(
    masked_img: np.typing.NDArray[Any], masker: MultiNiftiMasker
) -> Nifti1Image:
    """Convert a rasterized masked image back to an image."""
    return masker.inverse_transform(masked_img)


def basic_viz(
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: str = "t",
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
    stat: str = "t",
    **kwargs: dict[str, Any],
) -> StatMapView:
    """Simple interactive visualization."""
    return view_img(
        unmask(result[stat][idx, :], masker),
        **kwargs,
    )

import nanslice as ns
import numpy as np
from nilearn.maskers import MultiNiftiMasker

def nanslice_overlay(
    result: dict[str, np.ndarray],
    masker: MultiNiftiMasker,
    idx: int = 0,
    alpha_stat: str = "t",
    contour_stat: str = "logp_max_t",   # default changed
    alpha_lim: tuple[float, float] = (-2, 2),
    cmap: str = "RdYlBu_r",
    title: str = "Beta overlay",
) -> None:
    """
    Visualize beta coefficients as overlay, alpha controlled by t-statistics,
    contours from -log10 p-values (e.g., logp_max_t).
    """

    # Convert beta and alpha stat to NIfTI
    beta_img = masker.inverse_transform(result["beta_coef"][idx, :])
    alpha_img = masker.inverse_transform(result[alpha_stat][idx, :])

    # Set up contour data
    contour_img = None
    contour_level = None
    if contour_stat in result:
        contour_img = masker.inverse_transform(result[contour_stat][idx, :])
        # threshold corresponding to p < 0.05
        contour_level = -np.log10(0.05)  # ≈ 1.301

    # Create nanslice layers
    base_layer = ns.Layer(
        masker.mask_img_,
        cmap="gray",
        label="Mask/Background",
    )

    beta_layer = ns.Layer(
        beta_img,
        cmap=cmap,
        clim=(np.min(result["beta_coef"][idx, :]), np.max(result["beta_coef"][idx, :])),
        alpha=alpha_img,
        alpha_lim=alpha_lim,
        alpha_label=f"{alpha_stat.upper()}",
        label=f"Beta {idx}",
    )

    # Plot with nanslice
    if contour_img is not None and contour_level is not None:
        fig = three_plane(
            [base_layer, beta_layer],
            cbar=1,
            contour=(contour_img.get_fdata(), contour_level),
            title=title,
        )
    else:
        fig = three_plane([base_layer, beta_layer], cbar=1, title=title)

    return fig

