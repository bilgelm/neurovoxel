"""Visualize maps resulting from statistical analysis."""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownVariableType=false

from typing import Any

import numpy as np
from nilearn.maskers import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, view_img
from nilearn.plotting.displays._slicers import OrthoSlicer
from nilearn.plotting.html_stat_map import StatMapView


def basic_viz(
    result: dict[str, np.typing.NDArray[Any]],
    masker: MultiNiftiMasker,
    idx: int,
    stat: str = "t",
    **kwargs: dict[str, Any],
) -> OrthoSlicer:
    """Simple visualization."""
    return plot_stat_map(
        masker.inverse_transform(result[stat][idx, :]),
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
        masker.inverse_transform(result[stat][idx, :]),
        **kwargs,
    )
