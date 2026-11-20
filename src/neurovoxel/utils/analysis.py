"""Conduct statistical analysis."""

# pyright: reportMissingTypeStubs=false

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

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
from scipy.stats import f
from scipy.stats import t as student_t

from neurovoxel.utils import (
    MULTI_SES_OPTS,
    STANDARDIZATION_OPTS,
    ZERO_VOXEL_OPTS,
)
from neurovoxel.utils.formula import split_formula
from neurovoxel.utils.inference import fdr_log_p_values
from neurovoxel.utils.longitudinal import longitudinal_dependencies
from neurovoxel.utils.mixed import fit_mixed_model, prepare_mixed_design
from neurovoxel.utils.mixed_diagnostics import MixedDiagnostics
from neurovoxel.utils.results import (
    AnalysisResult,
    PermutedOLSResult,
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
        # Reports build preview images that are not used by NeuroVoxel.
        reports=False,
    )


def run_query(  # noqa: PLR0912, PLR0913, PLR0915, C901
    query: str,
    root: Path,
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
    correction: Literal["fwe", "fdr"] = "fwe",
    progress_callback: Callable[[int, int], None] | None = None,
    diagnostics_callback: Callable[[MixedDiagnostics], None] | None = None,
) -> tuple[AnalysisResult, pd.DataFrame]:
    """Run ordinary or longitudinal mixed-effects regression voxelwise.

    For OLS, each tested term is fitted independently, adjusting for non-tested
    terms, as in ``permuted_ols``. FDR adjusts two-sided Student t p-values
    across retained voxels separately for each tested term. Permutations
    and TFCE are only used for OLS FWE. Mixed formulas retain all sessions,
    fit fixed effects jointly, and apply FDR to asymptotic Wald p-values.
    ``progress_callback`` receives completed/total voxel counts for mixed fits.
    """
    if correction not in {"fwe", "fdr"}:
        msg = f"Unknown multiple comparison correction: {correction}"
        raise ValueError(msg)
    formula = split_formula(query)
    if formula.random is not None and correction != "fdr":
        msg = "Mixed-effects models require FDR correction (correction='fdr')"
        raise ValueError(msg)

    tbl_vars: set[str] = cast("Any", Formula)(
        f"{formula.lhs} ~ {formula.fixed}"
    ).rhs.required_variables
    # verify that each inference_term is in tbl_vars

    tbl_vars.update(["subject", "session"])  # pyright: ignore[reportUnknownMemberType]

    lhs, rhs = formula.lhs, formula.fixed
    if formula.random is not None:
        assert formula.group is not None  # noqa: S101
        tbl_vars.update(cast("Any", Formula)(formula.random).required_variables)
        tbl_vars.add(formula.group)
        tbl_vars = longitudinal_dependencies(tbl_vars, tbl)

    image_df = pd.DataFrame()
    for index, image in enumerate(images):
        image_df = pd.concat(
            [
                image_df,
                pd.DataFrame(
                    {
                        "subject": image.get_entities()["sub"],
                        "session": image.get_entities()["ses"],
                        lhs: root / image.path,
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
    if formula.random is None:
        tbl = select_sessions(tbl, handle_multiple_sessions, random_state)

    # bring subject and session columns to the front
    tbl = tbl[
        ["subject", "session"]
        + [col for col in tbl.columns if col not in ["subject", "session"]]
    ]

    mixed_design = None
    if formula.random is not None:
        mixed_design = prepare_mixed_design(
            formula, tbl, inference_terms, min_observations=MIN_N_OBS
        )
        tbl = mixed_design.table
        x_mat_mm = mixed_design.fixed
    else:
        x_mat_mm = model_matrix(rhs, tbl, na_action="ignore")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    x_mat = x_mat_mm.to_numpy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    predictor_names: np.typing.NDArray[Any] = np.asarray(x_mat_mm.columns)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    non_intercept = predictor_names != "Intercept"

    all_zero = np.all(x_mat == 0, axis=0)  # pyright: ignore[reportUnknownArgumentType]
    zero_terms = predictor_names[non_intercept & all_zero].tolist()

    # Cancel analysis if any predictor columns contain only zeros after removing
    # observations with missing predictors
    if zero_terms:
        message = (
            "Analysis stopped: these predictor columns contain only zeros: "
            + ", ".join(zero_terms)
        )
        raise ValueError(message)

    # Build y_mat from the image paths corresponding to the filtered `tbl` so
    # rows of y_mat align with rows of x_mat.
    y_mat = prepare_y_mat(
        tbl[lhs].to_numpy(),  # pyright: ignore[reportUnknownMemberType]
        masker,
        handle_zero_voxels,
        voxelwise_standardization,
    )

    if mixed_design is not None:
        return fit_mixed_model(
            mixed_design,
            y_mat,
            n_jobs,
            progress_callback=progress_callback,
            diagnostics_callback=diagnostics_callback,
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

    # Cancel analysis if any predictor columns contain only zeros after removing
    # observations with missing predictors
    all_zero = np.all(x_mat == 0, axis=0)  # pyright: ignore[reportUnknownArgumentType]
    zero_terms = predictor_names[non_intercept & all_zero].tolist()

    if zero_terms:
        raise ValueError(
            "Analysis stopped: these predictor columns contain only zeros "
            "after removing observations with missing predictors: "
            + ", ".join(zero_terms)
        )

    # ------------------------------
    # Compute OLS beta coefficients
    # ------------------------------
    ols_model = OLSModel(x_mat)  # pyright: ignore[reportUnknownArgumentType]
    ols_results = ols_model.fit(y_mat)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    beta_coef = ols_results.theta  # shape: (n_predictors, n_voxels)

    # MODEL DIAGNOSTICS

    r_square_results = ols_results.r_square  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] # shape: (n_voxels,)
    non_intercept = np.asarray(x_mat_mm.columns) != "Intercept"  # type: ignore  # noqa: PGH003

    # Unavailable diagnostics default to NaN.
    n_voxels = y_mat.shape[1]
    f_stat = np.full(n_voxels, np.nan, dtype=np.float64)
    f_pvalues = np.full(n_voxels, np.nan, dtype=np.float64)

    if non_intercept.any():
        contrast = np.eye(x_mat.shape[1])[non_intercept]  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        f_results = ols_results.Fcontrast(contrast)  # pyright: ignore[reportUnknownMemberType]

        f_stat = np.asarray(f_results.F).ravel()
        f_pvalues = f.sf(  # pyright: ignore[reportUnknownMemberType]
            f_stat,
            f_results.df_num,  # pyright: ignore[reportUnknownMemberType]
            f_results.df_den,  # pyright: ignore[reportUnknownMemberType]
        )

    # ------------------------------
    # Compute t-statistics and selected multiple comparison correction
    # ------------------------------
    is_inference_term = x_mat_mm.columns.isin(inference_terms)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    res = cast(
        "PermutedOLSResult",
        permuted_ols(  # pyright: ignore[reportUnknownVariableType]
            tested_vars=x_mat[:, is_inference_term],
            target_vars=y_mat,  # pyright: ignore[reportUnknownArgumentType]
            confounding_vars=x_mat[:, ~is_inference_term],
            n_perm=n_perm if correction == "fwe" else 0,
            n_jobs=n_jobs,
            random_state=random_state,
            masker=masker,
            tfce=tfce if correction == "fwe" else False,
            output_type="dict",
        ),
    )

    if correction == "fdr":
        assert "t" in res  # noqa: S101
        res["logp_fdr"] = fdr_log_p_values(
            ols_p_values(
                res["t"],
                x_mat[:, is_inference_term],  # pyright: ignore[reportUnknownArgumentType]
                x_mat[:, ~is_inference_term],  # pyright: ignore[reportUnknownArgumentType]
            )
        )

    res["tested_var_names"] = x_mat_mm.columns[is_inference_term].to_list()  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]

    # Attach OLS betas explicitly
    res["beta"] = beta_coef[is_inference_term, :]  # pyright: ignore[reportIndexIssue]
    res["r_square"] = r_square_results  # pyright: ignore[reportIndexIssue] # shape: (n_voxels,)
    res["f_stat"] = f_stat  # pyright: ignore[reportIndexIssue] # shape: (n_voxels,)
    res["f_pvalues"] = f_pvalues  # pyright: ignore[reportIndexIssue] # shape: (n_voxels,)

    # Include final table
    return res, tbl  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def select_sessions(
    tbl: pd.DataFrame, handle_multiple_sessions: str, random_state: int
) -> pd.DataFrame:
    """Select cross-sectional observations for ordinary linear regression."""
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
        chosen_idx: np.typing.NDArray[Any] = (
            tbl.groupby("subject", sort=False)[["session"]]  # pyright: ignore[reportUnknownMemberType]
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
    return tbl


def ols_p_values(
    t_scores: np.typing.NDArray[Any],
    tested_vars: np.typing.NDArray[Any],
    confounding_vars: np.typing.NDArray[Any],
) -> np.typing.NDArray[Any]:
    """Compute two-sided OLS p-values using residual degrees of freedom.

    The input t-scores must come from independently tested regressors, with
    the given confounds and an intercept, as returned by ``permuted_ols``.
    """
    confounds = np.column_stack(
        (np.ones(tested_vars.shape[0]), confounding_vars)
    )
    degrees_of_freedom = np.array(
        [
            tested_vars.shape[0]
            - np.linalg.matrix_rank(np.column_stack((confounds, term)))
            for term in tested_vars.T
        ]
    )
    if np.any(degrees_of_freedom <= 0):
        msg = "FDR inference requires positive residual degrees of freedom"
        raise ValueError(msg)

    return 2 * student_t.sf(  # pyright: ignore[reportUnknownMemberType]
        np.abs(t_scores), degrees_of_freedom[:, np.newaxis]
    )


def prepare_y_mat(
    lhs_values: np.typing.ArrayLike,
    masker: MultiNiftiMasker,
    handle_zero_voxels: str = ZERO_VOXEL_OPTS[0],  # pyright: ignore[reportUnknownParameterType, reportInvalidTypeForm])
    voxelwise_standardization: str = STANDARDIZATION_OPTS[0],
) -> np.typing.NDArray[Any]:
    """Exclude invalid voxels, update the masker, and standardize the data."""
    y_mat = cast(
        "np.typing.NDArray[Any]",
        np.vstack(masker.fit_transform(lhs_values)),  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownMemberType]
    )
    exclude_col = np.isnan(y_mat).any(axis=0)  # pyright: ignore[reportUnknownArgumentType]
    if handle_zero_voxels == "exclude":
        # Exclude voxels (columns) where any observation is 0.
        exclude_col = np.logical_or(exclude_col, (y_mat == 0).any(axis=0))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    keep_col = np.asarray(~exclude_col, dtype=bool)
    if not keep_col.any():
        msg = "No voxels remain after exclusion."
        raise ValueError(msg)

    if exclude_col.any():
        # Map the original column positions into 3D before updating the masker.
        updated_mask = masker.inverse_transform(keep_col.astype(np.uint8))  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        masker.set_params(mask_img=updated_mask).fit()  # pyright: ignore[reportUnknownMemberType]
        y_mat = y_mat[:, keep_col]

    # center (or scale) voxelwise values by computing a single mean
    # (and a single standard deviation) across all voxels and observations
    # - center: x - mean
    # - scale (z-score): (x - mean) / sd
    if voxelwise_standardization in ["center", "scale"]:
        y_mat -= np.mean(y_mat)
    if voxelwise_standardization == "scale":
        y_mat /= np.std(y_mat)

    return y_mat
