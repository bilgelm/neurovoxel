"""Conduct statistical analysis."""

# pyright: reportMissingTypeStubs=false

from typing import Any, NotRequired, TypedDict

import numpy as np
import pandas as pd
from bids.layout.models import BIDSImageFile
from formulaic import (
    Formula,
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)
from nibabel.nifti1 import Nifti1Image
from nilearn.glm import OLSModel
from nilearn.maskers import MultiNiftiMasker
from nilearn.mass_univariate import (
    permuted_ols,  # pyright: ignore[reportUnknownVariableType]
)

from neurovoxel.utils import (
    MULTI_SES_OPTS,
    STANDARDIZATION_OPTS,
    ZERO_VOXEL_OPTS,
)

MIN_N_OBS = 10  # minimum number of observations required for analysis


class PermutedOLSResult(TypedDict):
    """Permuted OLS result dictionary."""

    tested_var_names: list[str]
    beta: np.typing.NDArray[Any]
    t: np.typing.NDArray[Any]
    logp_max_t: np.typing.NDArray[Any]
    h0_max_t: np.typing.NDArray[Any]
    tfce: NotRequired[np.typing.NDArray[Any]]
    logp_max_tfce: NotRequired[np.typing.NDArray[Any]]
    h0_max_tfce: NotRequired[np.typing.NDArray[Any]]
    size: NotRequired[np.typing.NDArray[Any]]
    logp_max_size: NotRequired[np.typing.NDArray[Any]]
    h0_max_size: NotRequired[np.typing.NDArray[Any]]
    mass: NotRequired[np.typing.NDArray[Any]]
    logp_max_mass: NotRequired[np.typing.NDArray[Any]]
    h0_max_mass: NotRequired[np.typing.NDArray[Any]]


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
    inference_terms: set[str],
    tbl: pd.DataFrame,
    images: list[BIDSImageFile],
    masker: MultiNiftiMasker,
    n_perm: int,
    n_jobs: int,
    random_state: int,
    tfce: bool,
    handle_zero_voxels: str = ZERO_VOXEL_OPTS[0],  # pyright: ignore[reportUnknownParameterType, reportInvalidTypeForm]
    handle_multiple_sessions: str = MULTI_SES_OPTS[0],  # pyright: ignore[reportUnknownParameterType, reportInvalidTypeForm]
    voxelwise_standardization: str = STANDARDIZATION_OPTS[0],  # pyright: ignore[reportUnknownParameterType, reportInvalidTypeForm]
) -> tuple[PermutedOLSResult, pd.DataFrame]:
    """Run permuted OLS given a query, BIDS dataset, and tabular data file."""
    if "~" not in query:
        msg = "Invalid formula syntax"
        raise ValueError(msg)

    tbl_vars: set[str] = Formula(query).rhs.required_variables  # pyright: ignore[reportUnknownVariableType, reportAbstractUsage, reportUnknownMemberType, reportAttributeAccessIssue]
    # verify that each inference_term is in tbl_vars

    tbl_vars.update(["subject", "session"])  # pyright: ignore[reportUnknownMemberType]

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

    tbl = tbl[list(tbl_vars)].merge(  # pyright: ignore[reportUnknownArgumentType]
        image_df, on=["subject", "session"], how="inner"
    )

    tbl = tbl.sort_values(by=["subject", "session"]).reset_index(drop=True)
    # note: the way the data frame is made into a cross-sectional one does not
    # care about missingness in variables on the RHS.
    # a better implementation would be to take missingness into account.
    # to do this, we need a function that can take in RHS and return the
    # names of variables.
    if handle_multiple_sessions in ["first", "last"]:
        # Keep only the first/last row for each subject
        # (e.g., first/last session per subject)
        tbl = tbl.drop_duplicates(
            subset=["subject"],
            keep=handle_multiple_sessions,  # type: ignore[arg-type]
        ).reset_index(drop=True)
    elif handle_multiple_sessions == "random":
        # For each subject, sample a single row at random. Use the provided
        # random_state for reproducibility.
        rng = np.random.default_rng(random_state)
        # Group by subject and pick one index per group
        # FutureWarning: DataFrameGroupBy.apply operated on the grouping columns
        # This behavior is deprecated, and in a future version of pandas the
        # grouping columns will be excluded from the operation.
        # Either pass `include_groups=False` to exclude the groupings or
        # explicitly select the grouping columns after groupby to silence this
        # warning.
        chosen_idx = (
            tbl.groupby("subject", sort=False)  # pyright: ignore[reportUnknownMemberType]
            .apply(lambda g: rng.choice(g.index))
            .to_numpy()
        )
        tbl = tbl.loc[chosen_idx].reset_index(drop=True)
    elif handle_multiple_sessions == "all":
        # Do nothing, keep all visits
        pass
    else:
        msg = (
            f"{handle_multiple_sessions} is not a recognized option for "
            "handling multiple visits per subject"
        )
        raise ValueError(msg)

    # bring subject and session columns to the front
    tbl = tbl[
        ["subject", "session"]
        + [col for col in tbl.columns if col not in ["subject", "session"]]
    ]

    x_mat_mm = model_matrix(rhs, tbl, na_action="ignore")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    x_mat = x_mat_mm.to_numpy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    # Build y_mat from the image paths corresponding to the filtered `tbl` so
    # rows of y_mat align with rows of x_mat.
    y_mat = prepare_y_mat(
        np.vstack(masker.fit_transform(tbl[lhs].values)),  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType]
        handle_zero_voxels,
        voxelwise_standardization,
    )

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

    # ------------------------------
    # Compute OLS beta coefficients
    # ------------------------------
    ols_model = OLSModel(x_mat)  # pyright: ignore[reportUnknownArgumentType]
    ols_results = ols_model.fit(y_mat)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    beta_coef = ols_results.theta  # shape: (n_predictors, n_voxels)

    # ------------------------------
    # Run permutation-based inference
    # ------------------------------
    is_inference_term = x_mat_mm.columns.isin(inference_terms)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    res = permuted_ols(  # pyright: ignore[reportUnknownVariableType]
        tested_vars=x_mat[:, is_inference_term],
        target_vars=y_mat,  # pyright: ignore[reportUnknownArgumentType]
        confounding_vars=x_mat[:, ~is_inference_term],
        n_perm=n_perm,
        n_jobs=n_jobs,
        random_state=random_state,
        masker=masker,
        tfce=tfce,
        output_type="dict",
    )

    res["tested_var_names"] = x_mat_mm.columns[is_inference_term].to_list()  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]

    # Attach OLS betas explicitly
    res["beta"] = beta_coef[is_inference_term, :]  # pyright: ignore[reportIndexIssue]

    # Include final table
    return res, tbl  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def prepare_y_mat(
    y_mat: np.typing.NDArray[Any],
    handle_zero_voxels: str = ZERO_VOXEL_OPTS[0],  # pyright: ignore[reportUnknownParameterType, reportInvalidTypeForm])
    voxelwise_standardization: str = STANDARDIZATION_OPTS[0],
) -> np.typing.NDArray[Any]:
    """Handle NaN (and zero) voxels and perform z-scoring."""
    # There is a problem with the approach described below with "exclude"
    # Permutation test statistics are not computed correctly -
    # multiple comparison correction accounts for more voxels than needed,
    # yielding a more conservative result
    #
    # "exclude" voxels (columns) where any observation is NaN
    # Note about "excluding" voxels, throughout this script:
    # we don't actually exclude to not mess with masker; we just set all
    # observations for these voxels to 0, preventing any associations
    exclude_col = np.isnan(y_mat).any(axis=0)  # pyright: ignore[reportUnknownArgumentType]
    if handle_zero_voxels == "exclude":
        # "exclude" voxels (columns) where any observation is 0
        exclude_col = np.logical_or(exclude_col, (y_mat == 0).any(axis=0))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    y_mat[:, exclude_col] = 0  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # center (or scale) voxelwise values by computing a single mean
    # (and a single standard deviation) across all voxels and observations
    # compute stats only on non-excluded voxels
    # - center: x - mean
    # - scale (z-score): (x - mean) / sd
    if voxelwise_standardization in ["center", "scale"]:
        vox_mean = np.mean(y_mat[:, ~exclude_col])  # pyright: ignore[reportUnknownArgumentType]
        y_mat[:, ~exclude_col] -= vox_mean
    if voxelwise_standardization == "scale":
        vox_sd = np.std(y_mat[:, ~exclude_col])  # pyright: ignore[reportUnknownArgumentType]
        y_mat[:, ~exclude_col] /= vox_sd

    return y_mat
