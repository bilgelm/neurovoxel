"""Optimizer retries, failure details and persistent spatial diagnostic logs."""

# pyright: reportMissingTypeStubs=false

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import neurovoxel.utils.mixed as mod
from neurovoxel.utils.formula import split_formula
from neurovoxel.utils.mixed_diagnostics import (
    DIAGNOSTICS_FILENAME,
    SUMMARY_FILENAME,
    MixedDiagnostics,
    OptimizerAttempt,
    VoxelDiagnostic,
    save_mixed_diagnostics,
    summarize_mixed_diagnostics,
)


@pytest.mark.parametrize(
    "failure",
    [
        "LinAlgError",
        "ValueError",
        "FloatingPointError",
        "OverflowError",
        "nonconverged",
        "beta",
        "z",
        "p",
    ],
)
def test_optimizer_retries_with_fresh_model_and_keeps_warnings(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Keep unsuccessful attempts while retrying with a fresh model."""
    instances: list[object] = []
    methods: list[str] = []
    errors = {
        "LinAlgError": np.linalg.LinAlgError,
        "ValueError": ValueError,
        "FloatingPointError": FloatingPointError,
        "OverflowError": OverflowError,
    }

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            instances.append(self)

        def fit(self, *, method: str, **kwargs: object) -> SimpleNamespace:
            methods.append(method)
            assert kwargs == {
                "reml": True,
                "maxiter": mod.MIXED_MAXITER,
                "full_output": True,
                "disp": False,
            }
            first = len(methods) == 1
            if first:
                for _ in range(2):
                    warnings.warn(
                        "Covariance near a boundary",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if failure in errors:
                    msg = "deliberate optimizer failure"
                    raise errors[failure](msg)
            beta = np.array([1.0, 2.0])
            z = np.array([3.0, 4.0, 999.0])
            p = np.array([0.01, 0.02, 0.99])
            if first and failure in {"beta", "z", "p"}:
                {"beta": beta, "z": z, "p": p}[failure][0] = np.nan
            return SimpleNamespace(
                fe_params=beta,
                tvalues=z,
                pvalues=p,
                converged=not (first and failure == "nonconverged"),
                cov_re=np.diag([0.2, np.nan if first else 0.1]),
                scale=np.inf if first else 0.5,
                hist=[{"iterations": 7}],
            )

    monkeypatch.setattr(mod, "MixedLM", FakeModel)
    fit = mod._fit_voxel(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        np.ones(8), np.ones((8, 2)), np.ones((8, 2)), np.repeat([0, 1], 4)
    )
    assert fit.converged
    assert methods == ["lbfgs", "bfgs"]
    assert len(instances) == len({id(model) for model in instances})
    assert len(fit.attempts) == len(methods)
    first, accepted = fit.attempts
    assert first["warnings"] == [
        {
            "category": "RuntimeWarning",
            "message": "Covariance near a boundary",
            "count": 2,
        }
    ]
    if failure in errors:
        assert first["status"] == "exception"
        assert first.get("exception_type") == failure
        assert first.get("exception_message") == "deliberate optimizer failure"
        assert "deliberate optimizer failure" in first.get("traceback", "")
    else:
        expected = "nonconverged" if failure == "nonconverged" else "nonfinite"
        assert first["status"] == expected
        covariance = first.get("random_effect_covariance")
        assert covariance is not None
        assert covariance[1][1] is None
        assert first.get("residual_variance") is None
        if failure != "nonconverged":
            assert first.get("nonfinite_fields") == [failure]
    assert accepted["status"] == "success"
    expected_iterations = 7
    assert accepted.get("iterations") == expected_iterations
    assert accepted["warnings"] == []
    np.testing.assert_array_equal(fit.z, [3.0, 4.0])
    np.testing.assert_array_equal(fit.p, [0.01, 0.02])
    assert all(attempt["elapsed_seconds"] >= 0 for attempt in fit.attempts)
    json.dumps(list(fit.attempts), allow_nan=False)


def test_retry_exhaustion_keeps_every_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only assign failure placeholders after every optimizer has been tried."""
    methods: list[str] = []

    class BrokenModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self, *, method: str, **_kwargs: object) -> None:
            methods.append(method)
            msg = f"Singular matrix in {method}"
            raise np.linalg.LinAlgError(msg)

    monkeypatch.setattr(mod, "MixedLM", BrokenModel)
    fit = mod._fit_voxel(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        np.ones(8), np.ones((8, 2)), np.ones((8, 2)), np.repeat([0, 1], 4)
    )
    assert methods == list(mod.MIXED_OPTIMIZERS)
    assert not fit.converged
    assert np.isnan(fit.beta).all()
    assert np.isnan(fit.z).all()
    np.testing.assert_array_equal(fit.p, [1.0, 1.0])
    assert [a.get("exception_type") for a in fit.attempts] == [
        "LinAlgError"
    ] * len(methods)


@pytest.fixture
def diagnostics() -> MixedDiagnostics:
    """Include both a recovered voxel and an unsuccessful voxel."""
    failed: OptimizerAttempt = {
        "optimizer": "lbfgs",
        "status": "exception",
        "converged": None,
        "elapsed_seconds": 0.1,
        "warnings": [
            {"category": "RuntimeWarning", "message": "boundary", "count": 3}
        ],
        "exception_type": "LinAlgError",
        "exception_message": "singular",
    }
    accepted: OptimizerAttempt = {
        "optimizer": "bfgs",
        "status": "success",
        "converged": True,
        "elapsed_seconds": 0.2,
        "warnings": [],
    }
    voxels: list[VoxelDiagnostic] = [
        {
            "voxel_index": 0,
            "succeeded": True,
            "successful_optimizer": "bfgs",
            "attempts": [failed, accepted],
        },
        {
            "voxel_index": 1,
            "succeeded": False,
            "successful_optimizer": None,
            "attempts": [failed],
        },
    ]
    return summarize_mixed_diagnostics(
        voxels,
        ["Intercept", "age"],
        ["Intercept"],
        12,
        4,
        mod.MIXED_OPTIMIZERS,
        mod.MIXED_MAXITER,
    )


def test_summary_separates_recoveries_and_terminal_failures(
    diagnostics: MixedDiagnostics,
) -> None:
    """Count warning occurrences, not merely attempts that warned."""
    summary = diagnostics["summary"]
    assert summary["n_succeeded"] == 1
    assert summary["n_failed"] == 1
    assert summary["n_recovered_by_retry"] == 1
    assert summary["n_voxels_with_warnings"] == len(diagnostics["voxels"])
    assert summary["exception_counts_all_attempts"] == {"LinAlgError": 2}
    assert summary["final_failure_reasons"] == {"exception:LinAlgError": 1}
    assert summary["warning_counts_all_attempts"] == {"RuntimeWarning": 6}
    assert summary["successful_optimizer_counts"] == {"bfgs": 1}


def test_logs_include_voxel_coordinates(
    tmp_path: Path, diagnostics: MixedDiagnostics
) -> None:
    """Map compressed voxel indices back to the final fitted spatial grid."""
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 1, 0] = mask[1, 0, 1] = 1
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, -20.0, 30.0]
    save_mixed_diagnostics(tmp_path, diagnostics, Nifti1Image(mask, affine))
    summary = json.loads((tmp_path / SUMMARY_FILENAME).read_text())
    records = [
        json.loads(line)
        for line in (tmp_path / DIAGNOSTICS_FILENAME).read_text().splitlines()
    ]
    assert summary["n_failed"] == 1
    assert summary["grid_shape"] == [2, 2, 2]
    assert [r["voxel_ijk"] for r in records] == [[0, 1, 0], [1, 0, 1]]
    assert [r["voxel_xyz_mm"] for r in records] == [
        [10.0, -17.0, 30.0],
        [12.0, -20.0, 34.0],
    ]
    assert records[0]["attempts"][0]["exception_message"] == "singular"
    assert records[0]["successful_optimizer"] == "bfgs"


def test_all_failed_diagnostics_saved_before_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist details even when no valid inference result can be returned."""
    tbl = pd.DataFrame(
        {
            "subject": np.repeat(["a", "b", "c"], 4),
            "age": np.arange(12, dtype=float),
        }
    )
    design = mod.prepare_mixed_design(
        split_formula("gm ~ age + (1|subject)"), tbl, {"age"}
    )

    def broken(*_args: object) -> mod.VoxelFit:
        attempt: OptimizerAttempt = {
            "optimizer": "cg",
            "status": "nonconverged",
            "converged": False,
            "elapsed_seconds": 0.1,
            "warnings": [],
        }
        return mod.VoxelFit(
            np.full(2, np.nan),
            np.full(2, np.nan),
            np.ones(2),
            converged=False,
            attempts=(attempt,),
        )

    monkeypatch.setattr(mod, "_fit_voxel", broken)
    captured: list[MixedDiagnostics] = []

    def persist(diagnostics: MixedDiagnostics) -> None:
        captured.append(diagnostics)
        save_mixed_diagnostics(
            tmp_path,
            diagnostics,
            Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4)),
        )

    with pytest.raises(
        mod.MixedModelFitError, match="failed at every voxel"
    ) as caught:
        mod.fit_mixed_model(
            design, np.ones((12, 2)), 1, diagnostics_callback=persist
        )
    assert len(captured) == 1
    assert caught.value.diagnostics is captured[0]
    assert json.loads((tmp_path / SUMMARY_FILENAME).read_text())[
        "final_failure_reasons"
    ] == {"nonconverged": 2}
    assert len(
        (tmp_path / DIAGNOSTICS_FILENAME).read_text().splitlines()
    ) == len(captured[0]["voxels"])


def test_success_with_warning_stops_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warnings alone do not cause failure or unnecessary optimizer retries."""
    methods: list[str] = []

    class SuccessfulModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self, *, method: str, **_kwargs: object) -> SimpleNamespace:
            methods.append(method)
            warnings.warn("boundary estimate", RuntimeWarning, stacklevel=2)
            return SimpleNamespace(
                fe_params=np.ones(2),
                tvalues=np.ones(2),
                pvalues=np.full(2, 0.3),
                converged=True,
            )

    monkeypatch.setattr(mod, "MixedLM", SuccessfulModel)
    fit = mod._fit_voxel(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        np.ones(8), np.ones((8, 2)), np.ones((8, 2)), np.repeat([0, 1], 4)
    )
    assert methods == ["lbfgs"]
    assert fit.converged
    assert fit.attempts[0]["status"] == "success"
    assert fit.attempts[0]["warnings"][0]["message"] == "boundary estimate"
