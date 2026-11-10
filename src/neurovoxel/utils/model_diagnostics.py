"""Model diagnostic metrics for OLS."""

from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting  # pyright: ignore[reportMissingTypeStubs]
from nilearn.input_data import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)


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
    header.set_data_dtype(data.dtype) # pyright: ignore[reportUnknownMemberType]

    return Nifti1Image(data, img.affine, header)

# def significant_voxels_tables(
#     pdf: PdfPages,
#     r_square: np.typing.NDArray[np.float64],
#     masker: MultiNiftiMasker,
#     threshold: float,
#     rows_per_page: int = 45,
# ) -> None:
#     """Add tables of voxels with low R² to the PDF."""
#     values = np.asarray(r_square).ravel()

#     # Coordinates in the same order as the masked values
#     mask_img = masker.mask_img_ # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
#     if not isinstance(mask_img, Nifti1Image):
#         message = "masker.mask_img_ must be a Nifti1Image."
#         raise TypeError(message)

#     voxel_coordinates = np.argwhere(mask_img.get_fdata() != 0)

#     if values.size != len(voxel_coordinates):
#         message = "R² values do not match the fitted mask."
#         raise ValueError(message)

#     indices = np.flatnonzero(
#         np.isfinite(values) & (values < threshold)
#     )

#     total = len(indices)
#     n_pages = max(1, (total + rows_per_page - 1) // rows_per_page)

#     for page in range(n_pages):
#         selected = indices[
#             page * rows_per_page : (page + 1) * rows_per_page
#         ]

#         fig, ax = plt.subplots(figsize=(8.5, 11)) # pyright: ignore[reportUnknownMemberType]
#         ax.axis("off")
#         ax.set_title( # pyright: ignore[reportUnknownMemberType]
#             f"Voxels with R² < {threshold}\n"
#             f"{total:,} matching voxels — page {page + 1} of {n_pages}",
#             fontsize=12,
#             pad=15,
#         )

#         if selected.size:
#             coordinates = voxel_coordinates[selected]

#             rows = [
#                 [str(idx), str(i), str(j), str(k), f"{values[idx]:.6f}"]
#                 for idx, (i, j, k) in zip(selected, coordinates, strict=False)
#             ]

#             height = (len(rows) + 1) / (rows_per_page + 3)
#             table = ax.table( # pyright: ignore[reportUnknownMemberType]
#                 cellText=rows,
#                 colLabels=["Masked index", "i", "j", "k", "R²"],
#                 cellLoc="center",
#                 bbox=Bbox.from_bounds(0, 1 - height, 1, height),
#             )
#             table.auto_set_font_size(False)  # noqa: FBT003
#             table.set_fontsize(8)
#         else:
#             ax.text( # pyright: ignore[reportUnknownMemberType]
#                 0.5, 0.5, "No matching voxels.",
#                 ha="center", va="center",
#             )

#         pdf.savefig(fig) # pyright: ignore[reportUnknownMemberType]
#         plt.close(fig)

def generate_ols_results_pdf(
    stat_values: dict[str, np.typing.NDArray[np.float64]],
    masker: MultiNiftiMasker,
    bg_img: Nifti1Image,
    output_path: Path,
) -> None:
    """Generate a PDF containing OLS results.

    Include a brain map of R² values, tables of voxels with low R², and a
    histogram of R² values.
    """
    with PdfPages(output_path) as output_pdf:

        for img_key, values in stat_values.items():
            img = generate_stat_img(values, masker)
            fig = Figure(figsize=(10, 4))
            cmap = mpl.colormaps["viridis"].copy()
            cmap.set_under("black")

            # Change only the plotting copy; preserve NaNs in the original image
            data = img.get_fdata()
            plot_img = nib.Nifti1Image( # pyright: ignore[reportPrivateImportUsage]
                np.where(np.isnan(data), -1.0, data),
                img.affine,
            )

            plotting.plot_stat_map( # pyright: ignore[reportUnknownMemberType]
                plot_img,
                figure=fig,
                bg_img=bg_img,
                display_mode="ortho",
                cut_coords=(0, 0, 0),
                threshold=None, # pyright: ignore[reportArgumentType]
                cmap=cmap, # pyright: ignore[reportArgumentType]
                symmetric_cbar=False, # pyright: ignore[reportArgumentType]
                vmin=0,
                vmax=1,
                resampling_interpolation="nearest",
                title=f"Voxel-wise {img_key} map",
            )

            output_pdf.savefig(figure=fig, bbox_inches="tight") # pyright: ignore[reportUnknownMemberType]
            plt.close(fig)


        # Page 2: Histogram of R² values
        fig, ax = plt.subplots(figsize=(8, 5))  # pyright: ignore[reportUnknownMemberType]

        ax.hist(np.asarray( # pyright: ignore[reportUnknownMemberType]
            stat_values["r_square"], dtype=float).ravel(), bins=50
            )  # pyright: ignore[reportUnknownMemberType]
        ax.set(
            xlabel="R-squared",
            ylabel="Number of voxels",
            title="Distribution of voxel-wise R-squared",
        )

        output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)
