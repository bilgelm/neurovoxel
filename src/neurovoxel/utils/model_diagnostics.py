"""Model diagnostic metrics for OLS."""

from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting  # pyright: ignore[reportMissingTypeStubs]
from nilearn.input_data import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)

from neurovoxel.utils.viz import unmask


class StatValues(TypedDict):
    """Represents the statistical values for a single metric."""

    array: np.typing.NDArray[np.float64]
    vmax: float


def generate_ols_results_pdf(
    stat_values: dict[str, StatValues],
    masker: MultiNiftiMasker,
    bg_img: Nifti1Image,
    output_path: Path,
) -> None:
    """Generate a PDF containing OLS results.

    Include a brain map of R² values, tables of voxels with low R², and a
    histogram of R² values.
    """
    with PdfPages(output_path) as output_pdf:
        for key, value_dict in stat_values.items():
            values: np.typing.NDArray[np.float64] = value_dict["array"]
            img = unmask(values, masker, np.float64(np.nan))
            fig = Figure(figsize=(10, 4))
            cmap = mpl.colormaps["viridis"].copy()
            cmap.set_under("black")

            plotting.plot_stat_map(  # pyright: ignore[reportUnknownMemberType]
                img,
                figure=fig,
                bg_img=bg_img,
                display_mode="ortho",
                cut_coords=(0, 0, 0),
                threshold=None,  # pyright: ignore[reportArgumentType]
                cmap=cmap,  # pyright: ignore[reportArgumentType]
                symmetric_cbar=False,  # pyright: ignore[reportArgumentType]
                vmin=0,
                vmax=value_dict["vmax"],
                resampling_interpolation="nearest",
                title=f"Voxel-wise {key} map",
            )

            output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            plt.close(fig)

            # Page 2: Histogram of R² values
            fig, ax = plt.subplots(figsize=(8, 5))  # pyright: ignore[reportUnknownMemberType]

            ax.hist(  # pyright: ignore[reportUnknownMemberType]
                np.asarray(  # pyright: ignore[reportUnknownMemberType]
                    values, dtype=float
                ).ravel(),
                bins=50,
            )  # pyright: ignore[reportUnknownMemberType]
            ax.set(
                xlabel=f"{key}",
                ylabel="Number of voxels",
                title=f"Distribution of voxel-wise {key}",
            )

            output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            plt.close(fig)
