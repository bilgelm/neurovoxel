"""Model diagnostic metrics for OLS."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting  # pyright: ignore[reportMissingTypeStubs]
from nilearn.input_data import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)

if TYPE_CHECKING:
    from nilearn.plotting.displays import (  # pyright: ignore[reportMissingTypeStubs]
        OrthoSlicer,  # pyright: ignore[reportMissingTypeStubs]
    )


def generate_r_square_img(
    r_square: np.typing.NDArray[np.float64],
    masker: MultiNiftiMasker,
) -> Nifti1Image:
    """Generate a Nifti image of R² values using the provided masker."""
    values = np.asarray(r_square).ravel()

    if values.size != masker.n_elements_:
        message = (
            f"Got {values.size} R² values, but the masker "
            f"expects {masker.n_elements_} voxels."
        )
        raise ValueError(message)

    return cast("Nifti1Image", masker.inverse_transform(values)) # pyright: ignore[reportUnknownMemberType]

def add_low_r_squared_tables(
    pdf: PdfPages,
    r_square: np.typing.NDArray[np.float64],
    masker: MultiNiftiMasker,
    threshold: float = 0.4,
    rows_per_page: int = 45,
) -> None:
    """Add tables of voxels with low R² to the PDF."""
    values = np.asarray(r_square).ravel()

    # Coordinates in the same order as the masked values
    mask_img = masker.mask_img_ # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(mask_img, Nifti1Image):
        message = "masker.mask_img_ must be a Nifti1Image."
        raise TypeError(message)

    voxel_coordinates = np.argwhere(mask_img.get_fdata() != 0)

    if values.size != len(voxel_coordinates):
        message = "R² values do not match the fitted mask."
        raise ValueError(message)

    indices = np.flatnonzero(
        np.isfinite(values) & (values < threshold)
    )

    total = len(indices)
    n_pages = max(1, (total + rows_per_page - 1) // rows_per_page)

    for page in range(n_pages):
        selected = indices[
            page * rows_per_page : (page + 1) * rows_per_page
        ]

        fig, ax = plt.subplots(figsize=(8.5, 11)) # pyright: ignore[reportUnknownMemberType]
        ax.axis("off")
        ax.set_title( # pyright: ignore[reportUnknownMemberType]
            f"Voxels with R² < {threshold}\n"
            f"{total:,} matching voxels — page {page + 1} of {n_pages}",
            fontsize=12,
            pad=15,
        )

        if selected.size:
            coordinates = voxel_coordinates[selected]

            rows = [
                [str(idx), str(i), str(j), str(k), f"{values[idx]:.6f}"]
                for idx, (i, j, k) in zip(selected, coordinates, strict=False)
            ]

            height = (len(rows) + 1) / (rows_per_page + 3)
            table = ax.table( # pyright: ignore[reportUnknownMemberType]
                cellText=rows,
                colLabels=["Masked index", "i", "j", "k", "R²"],
                cellLoc="center",
                bbox=Bbox.from_bounds(0, 1 - height, 1, height),
            )
            table.auto_set_font_size(False)  # noqa: FBT003
            table.set_fontsize(8)
        else:
            ax.text( # pyright: ignore[reportUnknownMemberType]
                0.5, 0.5, "No matching voxels.",
                ha="center", va="center",
            )

        pdf.savefig(fig) # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)

def generate_ols_results_pdf(
    r_square: np.ndarray,
    masker: MultiNiftiMasker,
    r_square_img: Nifti1Image,
    bg_img: Nifti1Image,
    output_path: Path,
) -> None:
    """Generate a PDF containing OLS results.

    Include a brain map of R² values, tables of voxels with low R², and a
    histogram of R² values.
    """
    with PdfPages(output_path) as output_pdf:

        # Page 1: R² brain map
        fig = Figure(figsize=(10, 4))

        cast(
            "OrthoSlicer",
            plotting.plot_stat_map( # pyright: ignore[reportUnknownMemberType]
                r_square_img,
                figure=fig,
                bg_img=bg_img,
                display_mode="ortho",
                cut_coords=(0, 0, 0),
                threshold=None, # pyright: ignore[reportArgumentType]
                cmap="viridis",
                symmetric_cbar=False, # pyright: ignore[reportArgumentType]
                vmin=0,
                vmax=1,
                title="Voxel-wise R-squared",
            ),
        )

        output_pdf.savefig(figure=fig, bbox_inches="tight") # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)

        add_low_r_squared_tables(
            output_pdf,
            r_square,
            masker,
            threshold=0.4,
        )

        # Page 2: Histogram of R² values
        fig, ax = plt.subplots(figsize=(8, 5))  # pyright: ignore[reportUnknownMemberType]

        ax.hist(np.asarray(r_square, dtype=float).ravel(), bins=50)  # pyright: ignore[reportUnknownMemberType]
        ax.set(
            xlabel="R-squared",
            ylabel="Number of voxels",
            title="Distribution of voxel-wise R-squared",
        )

        output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)
