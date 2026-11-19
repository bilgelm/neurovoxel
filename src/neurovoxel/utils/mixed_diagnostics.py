"""Serializable optimizer diagnostics and spatially indexed fit logs."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image

DIAGNOSTICS_FILENAME = "lme_fit_diagnostics.jsonl"
SUMMARY_FILENAME = "lme_fit_summary.json"


class FitWarning(TypedDict):
    """A warning with repeated identical messages counted within an attempt."""

    category: str
    message: str
    count: int


class OptimizerAttempt(TypedDict):
    """One independent optimizer attempt, including unsuccessful inference."""

    optimizer: str
    status: Literal["success", "exception", "nonconverged", "nonfinite"]
    converged: bool | None
    elapsed_seconds: float
    warnings: list[FitWarning]
    exception_type: NotRequired[str]
    exception_message: NotRequired[str]
    traceback: NotRequired[str]
    nonfinite_fields: NotRequired[list[str]]
    iterations: NotRequired[int]
    random_effect_covariance: NotRequired[list[list[float | None]]]
    residual_variance: NotRequired[float | None]


class VoxelDiagnostic(TypedDict):
    """All attempts for a voxel in the final retained mask."""

    voxel_index: int
    succeeded: bool
    successful_optimizer: str | None
    attempts: list[OptimizerAttempt]


class MixedDiagnostics(TypedDict):
    """Fit summary and the full per-voxel attempt history."""

    summary: dict[str, Any]
    voxels: list[VoxelDiagnostic]


def summarize_mixed_diagnostics(  # noqa: PLR0913
    voxels: list[VoxelDiagnostic],
    fixed_names: list[str],
    random_names: list[str],
    n_observations: int,
    n_groups: int,
    optimizers: tuple[str, ...],
    maxiter: int,
) -> MixedDiagnostics:
    """Separate terminal failures, recovered retries and warning occurrences."""
    failures: Counter[str] = Counter()
    exceptions: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    warning_examples: dict[str, list[str]] = defaultdict(list)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    successful: Counter[str] = Counter()
    warned = recovered = succeeded = 0
    for voxel in voxels:
        succeeded += voxel["succeeded"]
        attempts = voxel["attempts"]
        warned += any(attempt["warnings"] for attempt in attempts)
        recovered += voxel["succeeded"] and len(attempts) > 1
        if voxel["successful_optimizer"] is not None:
            successful[voxel["successful_optimizer"]] += 1
        if not voxel["succeeded"]:
            final = attempts[-1] if attempts else None
            reason = final["status"] if final else "unknown"
            if final and "exception_type" in final:
                reason += f":{final['exception_type']}"
            failures[reason] += 1
        for attempt in attempts:
            outcomes[attempt["optimizer"]][attempt["status"]] += 1
            if "exception_type" in attempt:
                exceptions[attempt["exception_type"]] += 1
            for warning in attempt["warnings"]:
                category = warning["category"]
                warning_counts[category] += warning["count"]
                examples = warning_examples[category]
                if warning["message"] not in examples and len(examples) < 3:  # noqa: PLR2004
                    examples.append(warning["message"])
    return {
        "summary": {
            "n_voxels": len(voxels),
            "n_succeeded": succeeded,
            "n_failed": len(voxels) - succeeded,
            "n_recovered_by_retry": recovered,
            "n_voxels_with_warnings": warned,
            "final_failure_reasons": dict(failures),
            "exception_counts_all_attempts": dict(exceptions),
            "warning_counts_all_attempts": dict(warning_counts),
            "warning_examples": dict(warning_examples),
            "optimizer_outcomes": {
                key: dict(value) for key, value in outcomes.items()
            },
            "successful_optimizer_counts": dict(successful),
            "fixed_effect_names": fixed_names,
            "random_effect_names": random_names,
            "n_observations": n_observations,
            "n_groups": n_groups,
            "settings": {
                "reml": True,
                "optimizers": list(optimizers),
                "maxiter": maxiter,
            },
        },
        "voxels": voxels,
    }


def save_mixed_diagnostics(
    outputdir: Path, diagnostics: MixedDiagnostics, mask: "Nifti1Image"
) -> None:
    """Save one JSONL record per voxel, with IJK and world coordinates.

    NumPy boolean-mask order is the same C order used by Nilearn extraction.
    The mask must be the fitted mask after any voxel exclusions.
    """
    if mask.affine is None:
        msg = "Mixed-model diagnostics require a mask affine"
        raise ValueError(msg)
    affine = np.asarray(mask.affine, dtype=float)
    indices = np.argwhere(np.asarray(mask.dataobj, dtype=bool))
    if len(indices) != len(diagnostics["voxels"]):
        msg = "Mixed-model diagnostics do not match the fitted mask"
        raise ValueError(msg)
    outputdir.mkdir(parents=True, exist_ok=True)
    with (outputdir / DIAGNOSTICS_FILENAME).open("w") as stream:
        for voxel in diagnostics["voxels"]:
            ijk = indices[voxel["voxel_index"]]
            xyz = affine[:3, :3] @ ijk + affine[:3, 3]
            record = {
                **voxel,
                "voxel_ijk": ijk.tolist(),
                "voxel_xyz_mm": xyz.tolist(),
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {
        **diagnostics["summary"],
        "diagnostics_file": DIAGNOSTICS_FILENAME,
        "voxel_index_order": (
            "C order of nonzero voxels in fitted mask; zero based"
        ),
        "grid_shape": list(mask.shape),
        "grid_affine": affine.tolist(),
    }
    with (outputdir / SUMMARY_FILENAME).open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
