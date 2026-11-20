"""Reports for voxelwise ordinary and mixed model fit statistics."""

from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting  # pyright: ignore[reportMissingTypeStubs]
from nilearn.maskers import (  # pyright: ignore[reportMissingTypeStubs]
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
    *,
    mixed: bool = False,
) -> None:
    """Generate model fit maps and histograms (OLS-compatible entry point).

    Mixed reports identify the marginal and conditional R² and distinguish
    the asymptotic Wald F statistic from an OLS F test.
    """
    with PdfPages(output_path) as output_pdf:
        mixed = mixed or "r_square_conditional" in stat_values
        labels = {
            "r_square": "Marginal R² (fixed effects)" if mixed else "R²",
            "r_square_conditional": "Conditional R² (fixed + random effects)",
            "f_stat": "Asymptotic Wald F (df_den = infinity)"
            if mixed
            else "F statistic",
            "f_pvalues": "Overall fixed-effect p-value (uncorrected)"
            if mixed
            else "Overall F-test p-value (uncorrected)",
        }
        for key, value_dict in stat_values.items():
            label = labels.get(key, key)
            values: np.typing.NDArray[np.float64] = value_dict["array"]
            img = unmask(values, masker, np.float64(-1))
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
                title=f"Voxel-wise {label} map",
            )

            output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            plt.close(fig)

            # Page 2: Histogram of R² values
            fig, ax = plt.subplots(figsize=(8, 5))  # pyright: ignore[reportUnknownMemberType]

            finite_values = np.asarray(values, dtype=float).ravel()
            finite_values = finite_values[np.isfinite(finite_values)]
            ax.hist(finite_values, bins=50)  # pyright: ignore[reportUnknownMemberType]
            ax.set(
                xlabel=label,
                ylabel="Number of voxels",
                title=f"Distribution of voxel-wise {label}",
            )

            output_pdf.savefig(figure=fig, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            plt.close(fig)
