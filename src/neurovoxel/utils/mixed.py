"""Voxelwise longitudinal linear mixed-effects models."""

# pyright: reportMissingTypeStubs=false

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from formulaic import (
    Formula,
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)
from joblib import (
    Parallel,
    delayed,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.regression.mixed_linear_model import MixedLM

from neurovoxel.utils.formula import ModelFormula
from neurovoxel.utils.inference import fdr_log_p_values
from neurovoxel.utils.longitudinal import (
    LONGITUDINAL_TERMS,
    add_longitudinal_terms,
    longitudinal_dependencies,
)
from neurovoxel.utils.results import AnalysisResult

type VoxelFit = tuple[
    np.typing.NDArray[Any],
    np.typing.NDArray[Any],
    np.typing.NDArray[Any],
    bool,
    bool,
]


def _fit_voxel(
    y: np.typing.NDArray[Any],
    x: np.typing.NDArray[Any],
    random: np.typing.NDArray[Any],
    groups: np.typing.NDArray[Any],
) -> VoxelFit:
    """Fit one REML model, marking unsuccessful inference conservatively."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = cast("Any", MixedLM(y, x, groups, exog_re=random)).fit(
                reml=True,
                method=["lbfgs", "bfgs", "cg"],
                disp=False,
            )
            beta = np.asarray(fit.fe_params)
            z = np.asarray(fit.tvalues[: x.shape[1]])
            p = np.asarray(fit.pvalues[: x.shape[1]])
            if fit.converged and np.isfinite([beta, z, p]).all():
                return beta, z, p, True, bool(caught)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass
    missing = np.full(x.shape[1], np.nan)
    conservative = cast(
        "np.typing.NDArray[np.float64]",
        np.ones(x.shape[1], dtype=np.float64),
    )
    return missing, missing.copy(), conservative, False, bool(caught)


def _fit_indexed_voxel(
    index: int,
    y: np.typing.NDArray[Any],
    x: np.typing.NDArray[Any],
    random: np.typing.NDArray[Any],
    groups: np.typing.NDArray[Any],
) -> tuple[int, VoxelFit]:
    """Keep voxel identity when parallel fits finish out of order."""
    return index, _fit_voxel(y, x, random, groups)


def _fit_voxels(  # noqa: PLR0913
    y: np.typing.NDArray[Any],
    x: np.typing.NDArray[Any],
    z: np.typing.NDArray[Any],
    group_values: np.typing.NDArray[Any],
    n_jobs: int,
    progress_callback: Callable[[int, int], None] | None,
) -> list[VoxelFit]:
    """Collect completed fits on the calling thread, preserving voxel order."""
    total = y.shape[1]
    if progress_callback is not None:
        progress_callback(0, total)
    completed_fits = cast(
        "Iterable[tuple[int, VoxelFit]]",
        Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_fit_indexed_voxel)(index, column, x, z, group_values)
            for index, column in enumerate(y.T)
        ),
    )
    fits_by_voxel: dict[int, VoxelFit] = {}
    for index, fit in completed_fits:
        fits_by_voxel[index] = fit
        if progress_callback is not None:
            progress_callback(len(fits_by_voxel), total)
    return [fits_by_voxel[index] for index in range(total)]


@dataclass(frozen=True)
class MixedDesign:
    """Complete-case model matrices and their matching visit table."""

    table: pd.DataFrame
    fixed: pd.DataFrame
    random: pd.DataFrame
    groups: np.typing.NDArray[Any]
    selected: np.typing.NDArray[np.bool_]


def _complete_mixed_matrices(
    formula: ModelFormula, tbl: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute baselines until no further model-based exclusions remain."""
    assert formula.random is not None  # noqa: S101
    assert formula.group is not None  # noqa: S101
    required: set[str] = (
        cast("Any", Formula)(formula.fixed).required_variables
        | cast("Any", Formula)(formula.random).required_variables
        | {formula.group}
    )
    derived = bool(required & LONGITUDINAL_TERMS)
    required = longitudinal_dependencies(required, tbl)
    if derived:
        # An incomplete earlier visit must not determine any subject's baseline.
        tbl = tbl.loc[tbl[list(required)].notna().all(axis=1)].copy()
    if tbl.empty:
        msg = "Insufficient observations for mixed-effects inference"
        raise ValueError(msg)
    while True:
        if derived:
            tbl = add_longitudinal_terms(tbl)
        fixed = cast(
            "pd.DataFrame", model_matrix(formula.fixed, tbl, na_action="ignore")
        )
        random = cast(
            "pd.DataFrame",
            model_matrix(formula.random, tbl, na_action="ignore"),
        )
        # Source checks also catch missing categoricals encoded as zero dummies.
        valid = (
            np.asarray(tbl[list(required)].notna().all(axis=1))
            & ~np.asarray(fixed.isna().any(axis=1))
            & ~np.asarray(random.isna().any(axis=1))
            & np.asarray(tbl[formula.group].notna())
        )
        retained = tbl.loc[valid].reset_index(drop=True)
        if not derived or valid.all() or retained.empty:
            return (
                retained,
                fixed.loc[valid].reset_index(drop=True),
                random.loc[valid].reset_index(drop=True),
            )
        # Formula transforms can exclude additional visits. Update the origin
        # of time before rebuilding matrices, rather than retaining stale times.
        tbl = retained


def prepare_mixed_design(
    formula: ModelFormula,
    tbl: pd.DataFrame,
    inference_terms: set[str],
    *,
    min_observations: int = 10,
) -> MixedDesign:
    """Select and validate complete visits before any image preprocessing."""
    tbl, fixed, random = _complete_mixed_matrices(formula, tbl)
    assert formula.group is not None  # noqa: S101
    groups = tbl[formula.group]
    x = fixed.to_numpy(dtype=float)
    z = random.to_numpy(dtype=float)
    group_values = np.asarray(groups)
    selected = np.array(
        [col in inference_terms for col in fixed.columns], dtype=bool
    )
    if not selected.any():
        msg = "Select at least one fixed-effect term for inference"
        raise ValueError(msg)
    if x.shape[0] < min_observations or x.shape[0] <= x.shape[1]:
        msg = "Insufficient observations for mixed-effects inference"
        raise ValueError(msg)
    if not z.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        msg = "Mixed models require random effects and a full-rank fixed design"
        raise ValueError(msg)
    if np.linalg.matrix_rank(z) != z.shape[1]:
        msg = "Rank-deficient random effects design"
        raise ValueError(msg)
    counts = pd.Series(group_values).value_counts()
    if len(counts) < 2 or not (counts > 1).any():  # noqa: PLR2004
        msg = "Mixed models require multiple groups and repeated observations"
        raise ValueError(msg)
    return MixedDesign(
        table=tbl,
        fixed=fixed,
        random=random,
        groups=group_values,
        selected=selected,
    )


def fit_mixed_model(
    design: MixedDesign,
    y: np.typing.NDArray[Any],
    n_jobs: int,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[AnalysisResult, pd.DataFrame]:
    """Fit complete-case images against the already prepared mixed design.

    The rows of y must match design.table, including preprocessing restricted
    to those visits. Wald tests use the asymptotic normal distribution.
    Failed voxels receive p=1 before FDR, and NaN beta/z values.
    The optional callback receives (completed, total) voxel counts on the
    calling thread, including failed fits, beginning at zero.
    """
    x = design.fixed.to_numpy(dtype=float)
    z = design.random.to_numpy(dtype=float)
    group_values = design.groups
    selected = design.selected
    if y.shape[0] != len(design.table):
        msg = "Mixed-model images must match the complete-case visit table"
        raise ValueError(msg)
    fits = _fit_voxels(y, x, z, group_values, n_jobs, progress_callback)
    beta, scores, p_values, converged, warned = zip(*fits, strict=True)
    if not any(converged):
        msg = (
            "Mixed-effects inference failed at every voxel; "
            "check the model and data"
        )
        raise ValueError(msg)
    n_failed = len(converged) - sum(converged)
    if n_failed or any(warned):
        warnings.warn(
            f"Mixed-effects fits: {n_failed}/{len(fits)} voxels failed "
            "inference "
            f"(NaN beta/z, p=1); {sum(warned)} fits emitted "
            "numerical warnings. "
            "Inspect the converged maps and model specification.",
            RuntimeWarning,
            stacklevel=2,
        )
    result: AnalysisResult = {
        "tested_var_names": design.fixed.columns[selected].to_list(),
        "beta": np.asarray(beta).T[selected],
        "z": np.asarray(scores).T[selected],
        "logp_fdr": fdr_log_p_values(np.asarray(p_values).T[selected]),
        "converged": np.tile(converged, (int(selected.sum()), 1)).astype(float),
    }
    return result, design.table
