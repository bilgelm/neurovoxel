"""Conduct statistical analysis."""

import formulaic  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)
from nilearn.mass_univariate import (
    permuted_ols,  # pyright: ignore[reportMissingTypeStubs]
)
from pandas import DataFrame


def run_query(  # noqa: PLR0913
    query: str,
    tbl: DataFrame,
    image_paths: str | Nifti1Image | list[str | Nifti1Image],
    mask: str | Nifti1Image,
    smoothing_fwhm: float,
    vox_size: float,
    n_perm: int,
    n_jobs: int,
    random_state: int,
    tfce: bool,  # noqa: FBT001
) -> dict[str, np.ndarray]:
    """Run permuted OLS given a query, BIDS dataset, and tabular data file."""
    if "~" not in query:
        msg = "Invalid formula syntax"
        raise ValueError(msg)

    lhs, rhs = query.split("~")
    lhs = lhs.strip()
    rhs = rhs.strip()
    x_mat = formulaic.model_matrix(rhs, tbl)  # pyright: ignore[reportUnknownMemberType]

    masker = MultiNiftiMasker(
        mask,
        smoothing_fwhm=smoothing_fwhm,
        target_affine=np.eye(3) * vox_size,
        n_jobs=n_jobs,
    )
    y_mat = np.vstack(masker.fit_transform(image_paths))

    return permuted_ols(
        tested_vars=x_mat,
        target_vars=y_mat,
        n_perm=n_perm,
        n_jobs=n_jobs,
        random_state=random_state,
        masker=masker,
        tfce=tfce,
        output_type="dict",
    )
