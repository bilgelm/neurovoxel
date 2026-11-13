"""Test analysis module."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)

import neurovoxel.utils.analysis as mod

######################
# Tests for get_masker()
######################

SMOOTHING_FWHM = 6.0
N_PERM = 99


def make_mask_img() -> Nifti1Image:
    """Create and return a small binary mask image for tests."""
    return nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int8), np.eye(4))  # pyright: ignore[reportPrivateImportUsage]


@pytest.mark.parametrize("vox_size", [1.0, 2.0, 3.5])
def test_get_masker_passes_arguments_to_masker(
    monkeypatch: pytest.MonkeyPatch, vox_size: float
) -> None:
    """The masker should receive target geometry and smoothing settings."""
    calls = SimpleNamespace(args=None, kwargs=None)

    class FakeMultiNiftiMasker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.args, calls.kwargs = args, kwargs

    monkeypatch.setattr(mod, "MultiNiftiMasker", FakeMultiNiftiMasker)
    mask_img = make_mask_img()

    n_jobs = 4
    result = mod.get_masker(mask_img, SMOOTHING_FWHM, vox_size, n_jobs)

    assert isinstance(result, FakeMultiNiftiMasker)
    assert calls.args == (mask_img,)
    assert calls.kwargs["smoothing_fwhm"] == SMOOTHING_FWHM
    assert calls.kwargs["n_jobs"] == n_jobs
    np.testing.assert_array_equal(
        calls.kwargs["target_affine"], np.eye(3) * vox_size
    )


def test_get_masker_builds_real_masker() -> None:
    """Constructing a MultiNiftiMasker doesn't fit or load anything."""
    mask_img = make_mask_img()

    masker = mod.get_masker(
        mask_img, smoothing_fwhm=SMOOTHING_FWHM, vox_size=3.0, n_jobs=2
    )

    assert masker.mask_img is mask_img  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert masker.smoothing_fwhm == SMOOTHING_FWHM  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    n_jobs = 2
    assert masker.n_jobs == n_jobs  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    target_affine = masker.target_affine  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue,reportUnknownMemberType]
    assert isinstance(target_affine, np.ndarray)
    np.testing.assert_array_equal(target_affine, np.eye(3) * 3.0)  # pyright: ignore[reportUnknownArgumentType]


######################
# Tests for run_query
######################


def test_run_query_aligns_rows_and_selects_inference_term(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure query execution aligns rows and keeps the inference term.

    The selected inference term should remain available after aligning the
    rows in the query output.
    """
    tbl = pd.DataFrame(
        {
            "subject": ["01", "02", "03"],
            "session": ["1", "1", "1"],
            "age": [10.0, 20.0, np.nan],
        }
    )

    class FakeImage:
        def __init__(self, subject: str) -> None:
            self.subject = subject
            self.path = f"sub-{subject}.nii.gz"

        def get_entities(self) -> dict[str, str]:
            return {"sub": self.subject, "ses": "1"}

    images = [FakeImage(subject) for subject in tbl["subject"]]

    class FakeMasker:
        def fit_transform(self, paths: list[Path]) -> list[np.ndarray]:
            # Pretends to mask images and return their respective arrays.
            assert list(paths) == [tmp_path / image.path for image in images]
            return [
                np.array([[1.0, 2.0]]),
                np.array([[3.0, 4.0]]),
                np.array([[5.0, 6.0]]),
            ]  # fake results of masking

    class FakeOLSModel:
        def __init__(self, x: np.ndarray) -> None:
            np.testing.assert_array_equal(
                x,
                [
                    [1.0, 10.0],  # subject 01: intercept and age
                    [1.0, 20.0],
                ],  # subject 02: intercept and age
            )

        def fit(self, y: np.ndarray) -> SimpleNamespace:
            np.testing.assert_array_equal(y, [[1.0, 2.0], [3.0, 4.0]])
            return SimpleNamespace(
                theta=np.array(
                    [
                        [100.0, 101.0],  # intercept betas
                        [200.0, 201.0],  # age betas
                    ]
                )
            )

    def fake_permuted_ols(**kwargs: object) -> dict[str, np.ndarray]:
        np.testing.assert_array_equal(kwargs["tested_vars"], [[10.0], [20.0]])
        np.testing.assert_array_equal(
            kwargs["confounding_vars"], [[1.0], [1.0]]
        )
        np.testing.assert_array_equal(
            kwargs["target_vars"], [[1.0, 2.0], [3.0, 4.0]]
        )
        assert kwargs["n_perm"] == N_PERM
        return {"stat": np.array([7.0])}

    def fake_formula(_query: object) -> SimpleNamespace:
        return SimpleNamespace(rhs=SimpleNamespace(required_variables={"age"}))

    monkeypatch.setattr(mod, "Formula", fake_formula)

    def fake_model_matrix(
        _rhs: object, _data: object, **kwargs: object
    ) -> pd.DataFrame:
        # run_query must keep NaN rows here and drop them itself later
        assert kwargs["na_action"] == "ignore"
        return pd.DataFrame(
            {
                "Intercept": 1.0,
                "age": np.asarray(tbl["age"], dtype=np.float64),
            }
        )

    monkeypatch.setattr(mod, "model_matrix", fake_model_matrix)
    monkeypatch.setattr(mod, "OLSModel", FakeOLSModel)
    monkeypatch.setattr(mod, "permuted_ols", fake_permuted_ols)

    def fake_prepare_y_mat(
        y: list[Path], masker: FakeMasker, *_args: object
    ) -> np.ndarray:
        return np.vstack(masker.fit_transform(y))

    monkeypatch.setattr(mod, "prepare_y_mat", fake_prepare_y_mat)
    monkeypatch.setattr(mod, "MIN_N_OBS", 2)

    result, final_tbl = mod.run_query(
        query="image ~ age",
        root=tmp_path,
        inference_terms={"age"},
        tbl=tbl,
        images=images,  # type: ignore[arg-type]
        masker=FakeMasker(),  # type: ignore[arg-type]
        n_perm=N_PERM,
        n_jobs=1,
        random_state=42,
        tfce=False,
    )

    assert result["tested_var_names"] == ["age"]
    np.testing.assert_array_equal(result["beta"], [[200.0, 201.0]])
    expected_final_tbl_rows = 3
    assert len(final_tbl) == expected_final_tbl_rows


def test_run_query_invalid_formula(tmp_path: Path) -> None:
    """Ensure malformed formulas raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid formula syntax"):
        mod.run_query(
            query="image age",  # No ~
            root=tmp_path,
            inference_terms={"age"},
            tbl=pd.DataFrame(),
            images=[],
            masker=MultiNiftiMasker(),
            n_perm=99,
            n_jobs=1,
            random_state=42,
            tfce=False,
        )


@pytest.mark.parametrize(
    ("session_option", "expected_ages"),
    [
        ("first", [10.0, 20.0]),
        ("last", [11.0, 21.0]),
        ("all", [10.0, 11.0, 20.0, 21.0]),
        ("random", None),
    ],
)
def test_run_query_selects_session(  # noqa: C901 # still don't fully understand this function
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session_option: str,
    expected_ages: list[float] | None,
) -> None:
    """Verify session selection respects the configured multi-session policy."""
    tbl = pd.DataFrame(
        {
            "subject": ["01", "01", "02", "02"],
            "session": ["1", "2", "1", "2"],
            "age": [10.0, 11.0, 20.0, 21.0],
        }
    )

    class FakeImage:
        def __init__(self, subject: str, session: str) -> None:
            self.subject = subject
            self.session = session
            self.path = f"sub-{subject}_ses-{session}.nii.gz"

        def get_entities(self) -> dict[str, str]:
            return {"sub": self.subject, "ses": self.session}

    images = [
        FakeImage(str(row.subject), str(row.session))
        for row in tbl.itertuples()
    ]

    image_values = {
        tmp_path / image.path: age
        for image, age in zip(images, tbl["age"], strict=True)
    }

    class FakeMasker:
        def fit_transform(self, paths: list[Path]) -> np.ndarray:
            return np.array(
                [
                    [image_values[path], image_values[path] + 100]
                    for path in paths
                ]
            )

    class FakeOLSModel:
        def __init__(self, x: np.ndarray) -> None:
            pass

        def fit(self, y: np.ndarray) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(theta=np.zeros((2, 2)))

    def fake_permuted_ols(**kwargs: object) -> dict[str, object]:
        tested_vars = kwargs["tested_vars"]
        assert isinstance(tested_vars, np.ndarray)
        selected_ages = np.asarray(tested_vars[:, 0], dtype=float)

        if expected_ages is not None:
            np.testing.assert_array_equal(selected_ages, expected_ages)

        np.testing.assert_array_equal(
            kwargs["target_vars"],
            np.array([[age, age + 100] for age in selected_ages]),
        )
        return {}

    def _fake_formula(_query: str) -> SimpleNamespace:
        return SimpleNamespace(rhs=SimpleNamespace(required_variables={"age"}))

    monkeypatch.setattr(mod, "Formula", _fake_formula)

    def _mock_model_matrix(
        _rhs: object, data: pd.DataFrame, **kwargs: object
    ) -> pd.DataFrame:
        assert kwargs["na_action"] == "ignore"
        return pd.DataFrame(
            {
                "Intercept": 1.0,
                "age": np.asarray(data["age"], dtype=float),
            }
        )

    monkeypatch.setattr(mod, "model_matrix", _mock_model_matrix)
    monkeypatch.setattr(mod, "OLSModel", FakeOLSModel)
    monkeypatch.setattr(mod, "permuted_ols", fake_permuted_ols)

    def _prepare_y_mat(
        y: list[Path], masker: FakeMasker, *_args: object
    ) -> np.ndarray:
        return masker.fit_transform(y)

    monkeypatch.setattr(mod, "prepare_y_mat", _prepare_y_mat)
    monkeypatch.setattr(mod, "MIN_N_OBS", 2)

    def run_once() -> pd.DataFrame:
        _, final_tbl = mod.run_query(
            query="image ~ age",
            root=tmp_path,
            inference_terms={"age"},
            tbl=tbl,
            images=images,  # pyright: ignore[reportArgumentType]
            masker=FakeMasker(),  # pyright: ignore[reportArgumentType]
            n_perm=99,
            n_jobs=1,
            random_state=42,
            tfce=False,
            handle_multiple_sessions=session_option,
        )
        return final_tbl

    final_tbl = run_once()

    if session_option == "random":
        assert final_tbl["subject"].tolist() == ["01", "02"]
        assert final_tbl.iloc[0]["age"] in [10.0, 11.0]
        assert final_tbl.iloc[1]["age"] in [20.0, 21.0]

        # Running again with the same seed should make the same selections.
        pd.testing.assert_frame_equal(final_tbl, run_once())
        return

    assert final_tbl["age"].tolist() == expected_ages


def test_run_query_too_few_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that running a query with too few observations raises an error."""
    monkeypatch.setattr(mod, "MIN_N_OBS", 3)

    tbl = pd.DataFrame(
        {
            "subject": ["01", "02", "03"],
            "session": ["1", "1", "1"],
            "age": [10.0, np.nan, np.nan],
        }
    )

    class FakeImage:
        def __init__(self, subject: str) -> None:
            self.subject = subject
            self.path = f"sub-{subject}.nii.gz"

        def get_entities(self) -> dict[str, str]:
            return {"sub": self.subject, "ses": "1"}

    images = [FakeImage(subject) for subject in tbl["subject"]]

    class FakeMasker:
        def fit_transform(self, paths: list[Path]) -> np.ndarray:
            assert list(paths) == [tmp_path / image.path for image in images]
            return np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            )

    with pytest.raises(
        ValueError,
        match="This is not sufficient for statistical analysis",
    ):
        mod.run_query(
            query="image ~ age",
            root=tmp_path,
            inference_terms={"age"},
            tbl=tbl,
            images=images,  # pyright: ignore[reportArgumentType]
            masker=FakeMasker(),  # pyright: ignore[reportArgumentType]
            n_perm=99,
            n_jobs=1,
            random_state=42,
            tfce=False,
        )


@pytest.mark.parametrize("standardization", ["center", "scale"])
def test_run_query_beta_and_t_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    standardization: str,
) -> None:
    """Verify beta and t-statistic maps for centered and scaled image data."""
    age = np.array([10.0, 20.0, 30.0, 40.0])

    tbl = pd.DataFrame(
        {
            "subject": ["01", "02", "03", "04"],
            "session": ["1", "1", "1", "1"],
            "age": age,
        }
    )

    class FakeImage:
        def __init__(self, subject: str) -> None:
            self.subject = subject
            self.path = f"sub-{subject}.nii.gz"

        def get_entities(self) -> dict[str, str]:
            return {"sub": self.subject, "ses": "1"}

    images = [FakeImage(subject) for subject in tbl["subject"]]

    noise = np.array([1.0, -2.0, 1.0, 0.0])
    y = 5.0 + 2.0 * age + noise

    # Independently compute the reference OLS slope and t-statistic.
    design = np.column_stack([np.ones_like(age), age])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    degrees_of_freedom = len(y) - design.shape[1]
    residual_variance = (residuals @ residuals) / degrees_of_freedom
    slope_variance = residual_variance * np.linalg.inv(design.T @ design)[1, 1]
    expected_t = coefficients[1] / np.sqrt(slope_variance)

    expected_beta = coefficients[1]
    if standardization == "scale":
        expected_beta /= np.std(y)

    class FakeMasker:
        def fit_transform(self, _paths: object) -> np.ndarray:
            """Return a fresh image-data array for each call."""
            return y[:, None].copy()

    monkeypatch.setattr(mod, "MIN_N_OBS", 4)

    result, _ = mod.run_query(
        query="image ~ age",
        root=tmp_path,
        inference_terms={"age"},
        tbl=tbl,
        images=images,  # pyright: ignore[reportArgumentType]
        masker=FakeMasker(),  # pyright: ignore[reportArgumentType]
        n_perm=100,
        n_jobs=1,
        random_state=42,
        tfce=False,
        voxelwise_standardization=standardization,
    )

    assert "beta" in result
    np.testing.assert_allclose(
        result["beta"],
        [[expected_beta]],
        rtol=1e-5,
        atol=1e-5,
    )

    assert "t" in result
    np.testing.assert_allclose(
        result["t"],
        [[expected_t]],
        rtol=1e-5,
        atol=1e-5,
    )


###############################
# Tests for prepare_y_mat()
###############################


def make_preparation_inputs(
    y: np.ndarray,
) -> tuple[np.ndarray, MultiNiftiMasker]:
    """Represent matrix columns as spatial voxels in small subject images."""
    shape = (y.shape[1], 2, 2)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, 0, 0] = 1
    masker = MultiNiftiMasker(mask_img=Nifti1Image(mask, np.eye(4)))
    images: list[Nifti1Image] = []
    for row in y:
        data = np.zeros(shape)
        data[:, 0, 0] = row
        images.append(Nifti1Image(data, np.eye(4)))
    return np.array(images, dtype=object), masker


@pytest.mark.parametrize("zero_opt", ["include", "exclude"])
@pytest.mark.parametrize("std_opt", ["none", "center", "scale"])
def test_preparation_excludes_columns_and_preserves_mapping(
    zero_opt: str,
    std_opt: str,
) -> None:
    """Retained data and reconstructed statistics must use the same voxels."""
    y = np.array([[0.0, 1.0, 3.0, 4.0], [2.0, 4.0, 5.0, 6.0]])
    images, masker = make_preparation_inputs(y)
    result = mod.prepare_y_mat(images, masker, zero_opt, std_opt)
    keep = np.array([zero_opt == "include", True, True, True])
    expected = y[:, keep].copy()
    if std_opt in ["center", "scale"]:
        expected -= expected.mean()
    if std_opt == "scale":
        expected /= expected.std()
    np.testing.assert_allclose(result, expected)
    assert masker.n_elements_ == keep.sum()
    reconstructed_img = cast("Nifti1Image", masker.inverse_transform(result))  # pyright: ignore[reportUnknownMemberType]
    reconstructed = reconstructed_img.get_fdata()
    np.testing.assert_allclose(reconstructed[keep, 0, 0, :].T, result)
    np.testing.assert_array_equal(reconstructed[~keep, 0, 0, :], 0)


@pytest.mark.parametrize("zero_opt", ["include", "exclude"])
@pytest.mark.parametrize("std_opt", ["none", "center", "scale"])
def test_nan_columns_excluded(
    zero_opt: str,
    std_opt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop NaN columns whenever they occur in the extracted matrix."""
    y = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 5.0]])
    images, masker = make_preparation_inputs(y)
    masker.fit()  # pyright: ignore[reportUnknownMemberType]
    # Nilearn can replace NaNs during extraction; inject extracted NaNs to
    # exercise our filtering independently of its preprocessing behavior.
    calls: list[np.typing.ArrayLike] = []

    def fake_fit_transform(values: np.typing.ArrayLike) -> np.ndarray:
        calls.append(values)
        return y

    monkeypatch.setattr(masker, "fit_transform", fake_fit_transform)
    result = mod.prepare_y_mat(images, masker, zero_opt, std_opt)
    assert len(calls) == 1
    assert calls[0] is images
    expected = y[:, [0, 2]].copy()
    if std_opt in ["center", "scale"]:
        expected -= expected.mean()
    if std_opt == "scale":
        expected /= expected.std()
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(
        cast("Nifti1Image", masker.mask_img_).get_fdata()[:, 0, 0],  # pyright: ignore[reportUnknownMemberType]
        [1, 0, 1],
    )


def test_all_voxels_excluded_raises() -> None:
    """Fail clearly when no voxels remain for inference."""
    images, masker = make_preparation_inputs(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="No voxels remain after exclusion"):
        mod.prepare_y_mat(images, masker, "exclude", "none")


PrepareYMat = Callable[[np.ndarray, str, str], np.ndarray]


@pytest.fixture
def prepare_y_mat(monkeypatch: pytest.MonkeyPatch) -> PrepareYMat:
    """Exercise extracted-matrix cases with a real fitted spatial masker."""

    def prepare(y: np.ndarray, zero_opt: str, std_opt: str) -> np.ndarray:
        images, masker = make_preparation_inputs(y)
        masker.fit()  # pyright: ignore[reportUnknownMemberType]

        # Preserve NaNs that Nilearn may otherwise replace during extraction.
        def fake_fit_transform(_values: np.typing.ArrayLike) -> np.ndarray:
            return y.copy()

        monkeypatch.setattr(masker, "fit_transform", fake_fit_transform)
        return mod.prepare_y_mat(images, masker, zero_opt, std_opt)

    return prepare


def make_y() -> np.ndarray:
    """Return a fresh matrix for each preprocessing test."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------- no-op path


def test_no_standardization_leaves_clean_data_unchanged(
    prepare_y_mat: PrepareYMat,
) -> None:
    """Ensure the no-op path preserves an already-clean matrix."""
    result = prepare_y_mat(make_y(), "include", "none")

    np.testing.assert_array_equal(result, make_y())


def test_input_matrix_is_preserved(prepare_y_mat: PrepareYMat) -> None:
    """Image extraction should leave the source matrix unchanged."""
    y = make_y()

    result = prepare_y_mat(y, "include", "center")

    assert result is not y
    np.testing.assert_array_equal(y, make_y())
    np.testing.assert_allclose(result, make_y() - make_y().mean())


# ------------------------------------------------------------------ NaN voxels


def test_nan_voxel_is_excluded_for_all_observations(
    prepare_y_mat: PrepareYMat,
) -> None:
    """A NaN in any observation should exclude the whole voxel."""
    y = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 5.0]])

    result = prepare_y_mat(y, "include", "none")

    np.testing.assert_array_equal(result, [[1.0, 3.0], [2.0, 5.0]])


def test_no_nans_remain_after_processing(prepare_y_mat: PrepareYMat) -> None:
    """Any NaN values should be removed or masked after processing."""
    y = np.array([[np.nan, 1.0], [2.0, 3.0]])

    result = prepare_y_mat(y, "include", "scale")

    assert not np.isnan(result).any()


# ----------------------------------------------------------------- zero voxels


def test_zero_voxel_excluded_when_requested(prepare_y_mat: PrepareYMat) -> None:
    """A zero voxel should be removed when exclusion is requested."""
    y = np.array([[0.0, 1.0], [2.0, 3.0]])

    result = prepare_y_mat(y, "exclude", "none")

    np.testing.assert_array_equal(result, [[1.0], [3.0]])


def test_zero_voxel_kept_when_not_excluding(prepare_y_mat: PrepareYMat) -> None:
    """A zero voxel should be preserved when inclusion is requested."""
    y = np.array([[0.0, 1.0], [2.0, 3.0]])

    result = prepare_y_mat(y, "include", "none")

    np.testing.assert_array_equal(result, [[0.0, 1.0], [2.0, 3.0]])


def test_nan_voxels_excluded_regardless_of_zero_option(
    prepare_y_mat: PrepareYMat,
) -> None:
    """NaN voxels should be removed regardless of the zero filtering option."""
    for option in ("include", "exclude"):
        y = np.array([[np.nan, 1.0], [2.0, 3.0]])

        result = prepare_y_mat(y, option, "none")

        np.testing.assert_array_equal(result, [[1.0], [3.0]])


# --------------------------------------------------------------- standardizing


def test_center_subtracts_single_global_mean(
    prepare_y_mat: PrepareYMat,
) -> None:
    """Centering should subtract one global mean from every voxel value."""
    # global mean of all values = 2.5 (one mean, not one per column)
    result = prepare_y_mat(make_y(), "include", "center")

    np.testing.assert_allclose(result, [[-1.5, -0.5], [0.5, 1.5]])


def test_scale_gives_zero_mean_unit_sd(prepare_y_mat: PrepareYMat) -> None:
    """Scaling should standardize to zero mean and unit population SD."""
    result = prepare_y_mat(make_y(), "include", "scale")

    np.testing.assert_allclose(float(np.mean(result)), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(np.std(result)), 1.0, atol=1e-12)


def test_scale_matches_expected_values(prepare_y_mat: PrepareYMat) -> None:
    """Scaling should match the global z-score transform expected.

    from the data.
    """
    sd = np.sqrt(1.25)  # population sd (np.std default) of centered values

    result = prepare_y_mat(make_y(), "include", "scale")

    np.testing.assert_allclose(
        result, np.array([[-1.5, -0.5], [0.5, 1.5]]) / sd
    )


def test_scale_preserves_relative_structure(prepare_y_mat: PrepareYMat) -> None:
    """Scaling is a global shift and stretch, so correlations are unchanged."""
    y = np.array([[1.0, 5.0, 2.0], [4.0, 1.0, 9.0], [7.0, 3.0, 6.0]])
    expected_corr = np.corrcoef(y.ravel(), np.arange(9))[0, 1]

    result = prepare_y_mat(y.copy(), "include", "scale")

    actual_corr = np.corrcoef(result.ravel(), np.arange(9))[0, 1]
    np.testing.assert_allclose(
        actual_corr, expected_corr, rtol=1e-12, atol=1e-12
    )


# ----------------------------------------------- interaction: exclusion + stats


def test_excluded_voxels_remain_absent_after_centering(
    prepare_y_mat: PrepareYMat,
) -> None:
    """Centering should retain only the valid voxel columns."""
    y = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 5.0]])

    result = prepare_y_mat(y, "include", "center")

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[-1.75, 0.25], [-0.75, 2.25]])


def test_stats_computed_only_from_non_excluded_voxels(
    prepare_y_mat: PrepareYMat,
) -> None:
    """Statistics should ignore voxels excluded by a NaN mask."""
    y = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 5.0]])
    kept_mean = np.mean([1.0, 3.0, 2.0, 5.0])  # 2.75; ignores column 1

    result = prepare_y_mat(y, "include", "center")

    np.testing.assert_allclose(
        result, np.array([[1.0, 3.0], [2.0, 5.0]]) - kept_mean
    )


def test_scale_with_excluded_zero_voxel(prepare_y_mat: PrepareYMat) -> None:
    """Zero voxels excluded from scaling should not affect kept data."""
    y = np.array([[0.0, 1.0, 3.0], [2.0, 5.0, 7.0]])

    result = prepare_y_mat(y, "exclude", "scale")

    assert result.shape == (2, 2)
    kept = result
    assert np.mean(kept) == 0.0
    np.testing.assert_allclose(np.std(kept), 1.0)


# ------------------------------------------------------------------------ shape


@pytest.mark.parametrize("zero_opt", ["include", "exclude"])
@pytest.mark.parametrize("std_opt", ["none", "center", "scale"])
def test_output_shape_tracks_retained_voxels(
    zero_opt: str,
    std_opt: str,
    prepare_y_mat: PrepareYMat,
) -> None:
    """Preserve observations and drop only the excluded voxel columns."""
    y = np.array([[0.0, np.nan, 3.0, 4.0], [2.0, 4.0, 5.0, 6.0]])

    result = prepare_y_mat(y, zero_opt, std_opt)

    expected_columns = 2 if zero_opt == "exclude" else 3
    assert result.shape == (2, expected_columns)
