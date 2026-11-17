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
from nibabel.loadsave import (
    save as save_img,  # pyright: ignore[reportUnknownVariableType]
)
from nibabel.nifti1 import Nifti1Image
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

    def prepare_y(
        paths: np.typing.NDArray[Any], *_args: object
    ) -> np.typing.NDArray[np.float64]:
        indices = [int(Path(path).name.split(".")[0]) for path in paths]
        return y[indices].copy()

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
            mod.prepare_mixed_design(
                split_formula("image ~ age + (1|id)"), tbl, {"age"}
            ),
            y,
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
            mod.prepare_mixed_design(
                split_formula("image ~ age + (1|id)"), tbl, {"age"}
            ),
            y,
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
    tbl, _ = longitudinal
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
        mod.prepare_mixed_design(split_formula(query), tbl, terms)


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


@pytest.mark.parametrize("standardization", ["none", "center", "scale"])
@pytest.mark.parametrize("zero_option", ["keep", "exclude"])
def test_incomplete_visits_excluded_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
    standardization: str,
    zero_option: str,
) -> None:
    """Discarded images must not affect voxel exclusion or normalization."""
    source, values = longitudinal
    tbl = source.iloc[:15].copy()
    values = values[:15].copy()
    # Each reason for dropping a visit is independent. The missing-group
    # image would exclude every voxel; the other two corrupt normalization.
    tbl.loc[12, "age"] = np.nan
    tbl.loc[13, "time"] = np.nan
    tbl.loc[14, "id"] = np.nan
    values[12] = 1e9
    values[13] = -1e9
    values[14] = 0
    paths: list[Path] = []
    images: list[Any] = []
    for index, row in tbl.iterrows():
        path = tmp_path / f"visit-{index}.nii.gz"
        data = np.zeros((2, 2, 2))
        data[:, 0, 0] = values[int(str(index))]
        save_img(Nifti1Image(data, np.eye(4)), path)
        paths.append(path)
        images.append(
            SimpleNamespace(
                path=path.name,
                get_entities=lambda row=row: {
                    "sub": row.subject,
                    "ses": row.session,
                },
            )
        )
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[:, 0, 0] = 1
    masker = MultiNiftiMasker(
        mask_img=Nifti1Image(mask, np.eye(4)), reports=False
    )
    transform = cast("Any", masker).fit_transform
    loaded: list[Path] = []

    def extract(
        paths: np.typing.NDArray[Any],
    ) -> list[np.typing.NDArray[Any]]:
        loaded.extend(Path(path) for path in paths)
        return transform(paths)

    monkeypatch.setattr(masker, "fit_transform", extract)
    expected = values[:12].copy()
    if standardization in {"center", "scale"}:
        expected -= expected.mean()
    if standardization == "scale":
        expected /= expected.std()

    def fits(
        y: np.typing.NDArray[Any],
        x: np.typing.NDArray[Any],
        z: np.typing.NDArray[Any],
        groups: np.typing.NDArray[Any],
        *_args: object,
    ) -> list[mod.VoxelFit]:
        np.testing.assert_allclose(y, expected)
        np.testing.assert_allclose(x[:, 1], np.asarray(tbl.iloc[:12]["age"]))
        np.testing.assert_allclose(z[:, 1], np.asarray(tbl.iloc[:12]["time"]))
        np.testing.assert_array_equal(groups, np.asarray(tbl.iloc[:12]["id"]))
        return [
            (np.zeros(2), np.ones(2), np.full(2, 0.5), True, False) for _ in y.T
        ]

    monkeypatch.setattr(mod, "_fit_voxels", fits)
    result, used = analysis.run_query(
        "image ~ age + (1 + time | id)",
        tmp_path,
        {"age"},
        tbl.sample(frac=1, random_state=7),
        images[::-1],
        masker,
        99,
        1,
        42,
        tfce=False,
        correction="fdr",
        handle_zero_voxels=zero_option,
        voxelwise_standardization=standardization,
    )
    assert loaded == paths[:12]
    pd.testing.assert_frame_equal(
        used[["subject", "session"]], tbl.iloc[:12][["subject", "session"]]
    )
    assert result["beta"].shape == (1, 2)
    assert masker.n_elements_ == expected.shape[1]


def test_too_few_complete_visits_fail_before_image_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
) -> None:
    """Validate the joint complete-case count before reading any images."""
    tbl, _ = longitudinal
    tbl["id"] = np.nan
    images = [
        SimpleNamespace(
            path=f"{index}.nii.gz",
            get_entities=lambda row=row: {
                "sub": row.subject,
                "ses": row.session,
            },
        )
        for index, row in tbl.iterrows()
    ]

    def unexpected_preprocessing(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Images should not be read")

    monkeypatch.setattr(analysis, "prepare_y_mat", unexpected_preprocessing)
    with pytest.raises(ValueError, match="Insufficient observations"):
        analysis.run_query(
            "image ~ age + (1 + time | id)",
            tmp_path,
            {"age"},
            tbl,
            cast("Any", images),
            MultiNiftiMasker(),
            99,
            1,
            42,
            tfce=False,
            correction="fdr",
        )


def test_fit_requires_images_aligned_with_complete_design(
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
) -> None:
    """Reject mismatched image rows rather than fit misaligned visits."""
    tbl, y = longitudinal
    design = mod.prepare_mixed_design(
        split_formula("image ~ age + (1|id)"), tbl, {"age"}
    )
    with pytest.raises(ValueError, match="must match the complete-case"):
        mod.fit_mixed_model(design, y[:-1], 1)


@pytest.mark.parametrize("random_category", [False, True])
def test_missing_categorical_covariate_is_incomplete(
    longitudinal: tuple[pd.DataFrame, np.typing.NDArray[np.float64]],
    random_category: bool,
) -> None:
    """An all-zero dummy encoding must not hide a missing source covariate."""
    tbl, _ = longitudinal
    tbl["category"] = pd.Series(["a", "b"] * len(tbl)).iloc[: len(tbl)]
    tbl.loc[0, "category"] = None
    query = (
        "image ~ age + (1 + C(category) | id)"
        if random_category
        else "image ~ age + C(category) + (1 | id)"
    )
    design = mod.prepare_mixed_design(split_formula(query), tbl, {"age"})
    assert len(design.table) == len(tbl) - 1
    assert design.table["category"].notna().all()
    np.testing.assert_array_equal(design.groups, np.asarray(tbl["id"].iloc[1:]))
