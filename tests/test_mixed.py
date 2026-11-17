"""Longitudinal model fitting and correction against Statsmodels."""

# pyright: reportMissingTypeStubs=false

from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_config
from nilearn.maskers import MultiNiftiMasker
from statsmodels.regression.mixed_linear_model import MixedLM

import neurovoxel.utils.mixed as mod
from neurovoxel.utils import analysis
from neurovoxel.utils.formula import split_formula
from neurovoxel.utils.inference import fdr_log_p_values


@pytest.fixture
def longitudinal() -> tuple[pd.DataFrame, np.typing.NDArray[np.float64]]:
    """Generate repeated visits with random intercepts and slopes."""
    rng = np.random.default_rng(51)
    n_subjects, n_visits = 25, 5
    groups = np.repeat(np.arange(n_subjects), n_visits)
    time = np.tile(np.linspace(-1, 1, n_visits), n_subjects)
    age = time + np.repeat(rng.normal(size=n_subjects), n_visits)
    random = rng.multivariate_normal([0, 0], [[2, 0.2], [0.2, 0.6]], n_subjects)
    y = np.column_stack(
        [
            2
            + 0.7 * age
            + random[groups, 0]
            + time * random[groups, 1]
            + rng.normal(scale=0.5, size=len(groups))
            for _ in range(2)
        ]
    )
    tbl = pd.DataFrame(
        {
            "subject": [f"{i:02}" for i in groups],
            "session": np.tile(np.arange(n_visits).astype(str), n_subjects),
            "id": groups.astype(float),
            "age": age,
            "time": time,
        }
    )
    return tbl, y


@pytest.mark.parametrize("random", ["1", "1 + time", "0 + time"])
def test_longitudinal_matches_statsmodels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
    random: str,
) -> None:
    """Check effects, Wald scores, FDR, visit retention and row alignment."""
    tbl, y = longitudinal
    tbl.loc[0, "age"] = np.nan
    tbl.loc[6, "time"] = np.nan
    tbl.loc[12, "id"] = np.nan
    images = [
        SimpleNamespace(
            path=f"{i}.nii.gz",
            get_entities=lambda row=row: {
                "sub": row.subject,
                "ses": row.session,
            },
        )
        for i, row in tbl.iterrows()
    ]

    def prepare_y(*_args: object) -> np.typing.NDArray[np.float64]:
        return y.copy()

    monkeypatch.setattr(analysis, "prepare_y_mat", prepare_y)
    progress: list[tuple[int, int]] = []

    def update(completed: int, total: int) -> None:
        progress.append((completed, total))

    result, used = analysis.run_query(
        f"image ~ age + ({random} | id)",
        tmp_path,
        {"age", "Intercept"},
        tbl,
        cast("Any", images),
        MultiNiftiMasker(),
        99,
        1,
        42,
        tfce=True,
        correction="fdr",
        handle_multiple_sessions="first",
        progress_callback=update,
    )
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert "z" in result
    assert "converged" in result
    assert "logp_fdr" in result
    assert "t" not in result
    assert "logp_max_t" not in result
    required = ["age", "id"] + (["time"] if "time" in random else [])
    keep = tbl[required].notna().all(axis=1)
    expected_tbl = tbl.loc[keep]
    assert len(used) == len(expected_tbl)
    assert used["subject"].duplicated().any()
    expected_beta: list[Any] = []
    expected_z: list[Any] = []
    expected_p: list[Any] = []
    for column in y[keep].T:
        data = expected_tbl.assign(outcome=column)
        direct = (
            cast("Any", MixedLM)
            .from_formula(
                "outcome ~ age",
                data,
                groups=data["id"],
                re_formula=random,
            )
            .fit(reml=True, method=["lbfgs", "bfgs", "cg"], disp=False)
        )
        expected_beta.append(direct.fe_params)
        expected_z.append(direct.tvalues[:2])
        expected_p.append(direct.pvalues[:2])
    np.testing.assert_allclose(
        result["beta"], np.asarray(expected_beta).T, rtol=1e-6
    )
    np.testing.assert_allclose(result["z"], np.asarray(expected_z).T, rtol=1e-6)
    np.testing.assert_allclose(
        result["logp_fdr"],
        fdr_log_p_values(np.asarray(expected_p).T),
        rtol=1e-6,
    )
    assert result["converged"].all()


def test_mixed_rejects_fwe(tmp_path: Path) -> None:
    """Do not apply independent-observation permutations to repeated visits."""
    with pytest.raises(ValueError, match="require FDR"):
        analysis.run_query(
            "image ~ age + (1|subject)",
            tmp_path,
            {"age"},
            pd.DataFrame(),
            [],
            MultiNiftiMasker(),
            99,
            1,
            42,
            tfce=False,
        )


def test_failed_voxel_is_conservative(
    monkeypatch: pytest.MonkeyPatch,
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
) -> None:
    """Keep failed voxels in the FDR family with p=1 and flag their maps."""
    tbl, y = longitudinal
    calls = 0

    def fake_fit(*_args: object) -> mod.VoxelFit:
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                np.array([2.0, 0.7]),
                np.array([2.0, 3.0]),
                np.array([0.04, 0.01]),
                True,
                False,
            )
        return np.full(2, np.nan), np.full(2, np.nan), np.ones(2), False, True

    monkeypatch.setattr(mod, "_fit_voxel", fake_fit)
    with pytest.warns(RuntimeWarning, match="1/2 voxels failed"):
        result, _ = mod.fit_mixed_model(
            split_formula("image ~ age + (1|id)"),
            tbl,
            y,
            np.ones(len(tbl), dtype=bool),
            {"age"},
            1,
        )
    assert "z" in result
    assert "converged" in result
    assert "logp_fdr" in result
    np.testing.assert_array_equal(result["converged"], [[1, 0]])
    assert np.isnan(result["z"][0, 1])
    np.testing.assert_allclose(10.0 ** -result["logp_fdr"], [[0.02, 1]])


def test_all_voxels_fail(
    monkeypatch: pytest.MonkeyPatch,
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
) -> None:
    """Stop with a clear error when no voxel supports valid inference."""
    tbl, y = longitudinal
    fit = SimpleNamespace(
        converged=False,
        fe_params=np.ones(2),
        tvalues=np.ones(2),
        pvalues=np.ones(2),
    )

    def fake_fit(**_kwargs: object) -> SimpleNamespace:
        return fit

    def fake_model(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(fit=fake_fit)

    monkeypatch.setattr(mod, "MixedLM", fake_model)
    with pytest.raises(ValueError, match="failed at every voxel"):
        mod.fit_mixed_model(
            split_formula("image ~ age + (1|id)"),
            tbl,
            y,
            np.ones(len(tbl), dtype=bool),
            {"age"},
            1,
        )


@pytest.mark.parametrize(
    "case",
    [
        "one_group",
        "no_repeats",
        "missing",
        "no_fixed",
        "no_random",
        "rank_fixed",
        "rank_random",
    ],
)
def test_mixed_invalid_design(
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
    case: str,
) -> None:
    """Reject unusable or deficient designs before voxel fitting."""
    tbl, y = longitudinal
    query = "image ~ age + (1|id)"
    terms = {"age"}
    if case == "one_group":
        tbl["id"] = 1
    elif case == "no_repeats":
        tbl["id"] = np.arange(len(tbl))
    elif case == "missing":
        tbl["id"] = np.nan
    elif case == "no_fixed":
        terms = set[str]()
    elif case == "no_random":
        query = "image ~ age + (0|id)"
    elif case == "rank_fixed":
        tbl["age"] = 1.0
    else:
        tbl["time"] = 1.0
        query = "image ~ age + (1 + time|id)"
    with pytest.raises(ValueError, match="require|Insufficient|Select|[Rr]ank"):
        mod.fit_mixed_model(
            split_formula(query),
            tbl,
            y,
            np.ones(len(tbl), dtype=bool),
            terms,
            1,
        )


def test_parallel_progress_tracks_completions_and_preserves_voxel_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report later completed voxels while an earlier fit is still running."""
    release_first = Event()
    finished: list[int] = []
    updates: list[tuple[int, int]] = []
    caller_thread = get_ident()

    def fit(y: np.typing.NDArray[Any], *_args: object) -> mod.VoxelFit:
        voxel = int(y[0])
        if voxel == 0 and not release_first.wait(timeout=5):
            msg = "Progress did not report the completed second voxel"
            raise RuntimeError(msg)
        finished.append(voxel)
        value = np.array([float(voxel)])
        return value, value.copy(), np.ones(1), True, False

    def update(completed: int, total: int) -> None:
        assert get_ident() == caller_thread
        updates.append((completed, total))
        if completed == 1:
            assert finished == [1]
            release_first.set()

    monkeypatch.setattr(mod, "_fit_voxel", fit)
    with parallel_config(backend="threading"):
        results = mod._fit_voxels(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            np.array([[0.0, 1.0]]),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.array(["subject"]),
            2,
            update,
        )
    assert updates == [(0, 2), (1, 2), (2, 2)]
    assert finished == [1, 0]
    assert [float(result[0][0]) for result in results] == [0.0, 1.0]
