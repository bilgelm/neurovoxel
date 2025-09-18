"""Conduct statistical analysis."""

# pyright: reportMissingTypeStubs=false

from typing import Any, Literal

import numpy as np
import pandas as pd
from bids.layout.models import BIDSImageFile
from formulaic import (
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import MultiNiftiMasker
from nilearn.mass_univariate import (
    permuted_ols,  # pyright: ignore[reportUnknownVariableType]
)

MIN_N_OBS = 10  # minimum number of observations required for analysis


def get_masker(
    mask: str | Nifti1Image,
    smoothing_fwhm: float,
    vox_size: float,
    n_jobs: int,
) -> MultiNiftiMasker:
    """Get masker."""
    return MultiNiftiMasker(
        mask,
        smoothing_fwhm=smoothing_fwhm,
        target_affine=np.eye(3) * vox_size,
        n_jobs=n_jobs,
    )


def run_query(  # noqa: PLR0913
    query: str,
    tbl: pd.DataFrame,
    images: list[BIDSImageFile],
    masker: MultiNiftiMasker,
    n_perm: int,
    n_jobs: int,
    random_state: int,
    tfce: bool,
    handle_zero_voxels: Literal["keep", "exclude"] = "keep",
) -> dict[str, np.typing.NDArray[Any]]:
    """Run permuted OLS given a query, BIDS dataset, and tabular data file."""
    if "~" not in query:
        msg = "Invalid formula syntax"
        raise ValueError(msg)

    lhs, rhs = query.split("~")
    lhs = lhs.strip()
    rhs = rhs.strip()

    image_df = pd.DataFrame()
    for index, image in enumerate(images):
        image_df = pd.concat(
            [
                image_df,
                pd.DataFrame(
                    {
                        "subject": image.entities["subject"],
                        "session": image.entities["session"],
                        lhs: image.path,
                    },
                    index=[index],
                ),
            ]
        )

    tbl = tbl.merge(image_df, on=["subject", "session"], how="inner")

    x_mat = model_matrix(rhs, tbl, na_action="ignore").to_numpy()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    y_mat = np.vstack(masker.fit_transform(image_df[lhs].values))  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]

    # exclude voxels (columns) where any observation is NaN
    y_mat = y_mat[:, ~np.isnan(y_mat).any(axis=0)]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if handle_zero_voxels == "exclude":
        # exclude voxels (columns) where any observation is 0
        y_mat = y_mat[:, ~(y_mat == 0).any(axis=0)]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # standardize image values?: y_mat = (y_mat - y_mat.mean()) / y_mat.std()

    # handle missing independent variables by excluding observations
    # Remove rows from x_mat and y_mat where x_mat contains any missing values
    valid_rows = ~np.isnan(x_mat).any(axis=1)  # pyright: ignore[reportUnknownArgumentType]

    n_valid_rows = valid_rows.sum()
    if n_valid_rows < MIN_N_OBS:
        msg = (
            f"For the specified query {query}, there are only {n_valid_rows} "
            "observations in the dataset without any missingness. "
            "This is not sufficient for statistical analysis."
        )
        raise ValueError(msg)

    x_mat = x_mat[valid_rows, :]  # pyright: ignore[reportUnknownVariableType]
    y_mat = y_mat[valid_rows, :]  # pyright: ignore[reportUnknownVariableType]

    return permuted_ols(  # pyright: ignore[reportUnknownVariableType, reportReturnType]
        tested_vars=x_mat,
        target_vars=y_mat,  # pyright: ignore[reportUnknownArgumentType]
        n_perm=n_perm,
        n_jobs=n_jobs,
        random_state=random_state,
        masker=masker,
        tfce=tfce,
        output_type="dict",
    )
