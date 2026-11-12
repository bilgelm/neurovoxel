"""Model diagnostic metrics for OLS."""

from pathlib import Path
from typing import TypedDict, cast

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


class StatValues(TypedDict):
    """Represents the statistical values for a single metric."""

    array: np.typing.NDArray[np.float64]
    vmax: float


def generate_stat_img(
    voxel_data: np.typing.NDArray[np.float64],
    masker: MultiNiftiMasker,
) -> Nifti1Image:
    """Reconstruct statistical values, setting voxels outside mask to NaN.

    This only works for non negative values.
    """
    values = np.asarray(voxel_data).ravel()

    if values.size != masker.n_elements_:
        message = (
            f"Got {values.size} values, but the masker "
            f"expects {masker.n_elements_} voxels."
        )
        raise ValueError(message)

    img: Nifti1Image = cast(
        "Nifti1Image",
        masker.inverse_transform(values),  # pyright: ignore[reportUnknownMemberType]
    )
    data = img.get_fdata()

    mask_img = masker.mask_img_  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(mask_img, Nifti1Image):
        message = "masker.mask_img_ must be a Nifti1Image."
        raise TypeError(message)

    outside_mask: np.typing.NDArray[np.bool_] = mask_img.get_fdata() == 0
    data[outside_mask] = np.nan

    header = img.header.copy()
    header.set_data_dtype(data.dtype)  # pyright: ignore[reportUnknownMemberType]

    return Nifti1Image(data, img.affine, header)


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
            img = generate_stat_img(values, masker)
            fig = Figure(figsize=(10, 4))
            cmap = mpl.colormaps["viridis"].copy()
            cmap.set_under("black")

            # Change only the plotting copy; preserve NaNs in the original image
            data = img.get_fdata()
            plot_img = nib.Nifti1Image(  # pyright: ignore[reportPrivateImportUsage]
                np.where(np.isnan(data), -1.0, data),
                img.affine,
            )

            plotting.plot_stat_map(  # pyright: ignore[reportUnknownMemberType]
                plot_img,
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
