"""Unit tests for the viz module."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, call

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)
from nilearn.plotting.displays import (  # pyright: ignore[reportMissingTypeStubs]
    OrthoSlicer,  # pyright: ignore[reportMissingTypeStubs]
)
from nilearn.plotting.html_stat_map import (  # pyright: ignore[reportMissingTypeStubs]
    StatMapView,  # pyright: ignore[reportMissingTypeStubs]
)

import neurovoxel.utils.viz as mod

MultiNiftiMaskerType = Any


# test for unmask function


def test_unmask() -> None:
    """Verify unmask delegates to the masker and returns its image."""
    masked_img: np.typing.NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
    expected_image: Nifti1Image = Nifti1Image(
        np.zeros((3, 3, 3)),
        affine=np.eye(4),
    )
    masker: Mock = Mock(spec=MultiNiftiMasker)
    masker.inverse_transform.return_value = expected_image

    result: Nifti1Image = mod.unmask(masked_img, masker)

    masker.inverse_transform.assert_called_once_with(masked_img)
    assert result is expected_image


# test for basic_viz function


def test_basic_viz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot the correct row and forward plotting options."""
    result: mod.PermutedOLSResult = cast(
        "mod.PermutedOLSResult",
        {
            "tested_var_names": ["age", "sex"],
            "t": np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        },
    )
    masker: MultiNiftiMaskerType = cast(
        "MultiNiftiMaskerType",
        Mock(),
    )
    image: Nifti1Image = Nifti1Image(
        np.zeros((3, 3, 3)),
        affine=np.eye(4),
    )
    expected_plot = cast("OrthoSlicer", Mock(spec=OrthoSlicer))

    unmask_mock: Mock = Mock(return_value=image)
    plot_mock: Mock = Mock(return_value=expected_plot)
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "plot_stat_map", plot_mock)

    actual_plot: OrthoSlicer = mod.basic_viz(
        result,
        masker,
        term="sex",
        stat="t",
        title={"title": "Sex effect"},
    )

    unmask_mock.assert_called_once()
    np.testing.assert_array_equal(
        unmask_mock.call_args.args[0],
        [4.0, 5.0, 6.0],
    )
    assert unmask_mock.call_args.args[1] is masker
    plot_mock.assert_called_once_with(image, title={"title": "Sex effect"})
    assert actual_plot is expected_plot


def test_basic_viz_missing_statistic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a statistic that is absent from the results."""
    result: mod.PermutedOLSResult = cast(
        "mod.PermutedOLSResult",
        {"tested_var_names": ["age"]},
    )
    masker: MultiNiftiMaskerType = cast(
        "MultiNiftiMaskerType",
        Mock(),
    )
    unmask_mock: Mock = Mock()
    plot_mock: Mock = Mock()
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "plot_stat_map", plot_mock)

    with pytest.raises(KeyError, match="t is not in results"):
        mod.basic_viz(result, masker, term="age", stat="t")

    unmask_mock.assert_not_called()
    plot_mock.assert_not_called()


# Basic Interactive Viz test


def test_basic_interactive_viz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the correct term and return the interactive view."""
    result = cast(
        "mod.PermutedOLSResult",
        {
            "tested_var_names": ["age", "sex"],
            "t": np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        },
    )
    masker: MultiNiftiMaskerType = cast(
        "MultiNiftiMaskerType",
        Mock(spec="nilearn.maskers.MultiNiftiMasker"),
    )
    image: Nifti1Image = Nifti1Image(
        np.zeros((3, 3, 3)),
        affine=np.eye(4),
    )
    expected_view: StatMapView = cast(
        "StatMapView",
        Mock(spec=StatMapView),
    )

    unmask_mock: Mock = Mock(return_value=image)
    view_mock: Mock = Mock(return_value=expected_view)
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "view_img", view_mock)

    actual_view: StatMapView = mod.basic_interactive_viz(
        result,
        masker,
        term="sex",
        stat="t",
        title={"title": "Sex effect"},
        opacity={"opacity": 0.5},
    )

    unmask_mock.assert_called_once()
    np.testing.assert_array_equal(
        unmask_mock.call_args.args[0],
        [4.0, 5.0, 6.0],
    )
    assert unmask_mock.call_args.args[1] is masker
    view_mock.assert_called_once_with(
        image,
        title={"title": "Sex effect"},
        opacity={"opacity": 0.5},
    )
    assert actual_view is expected_view


def test_basic_interactive_viz_missing_statistic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a statistic that is absent from the results."""
    result = cast(
        "mod.PermutedOLSResult",
        {"tested_var_names": ["age"]},
    )
    masker: MultiNiftiMaskerType = cast(
        "MultiNiftiMaskerType",
        Mock(spec="nilearn.maskers.MultiNiftiMasker"),
    )

    unmask_mock: Mock = Mock()
    view_mock: Mock = Mock()
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "view_img", view_mock)

    with pytest.raises(KeyError, match="t is not in results"):
        mod.basic_interactive_viz(result, masker, term="age", stat="t")

    unmask_mock.assert_not_called()
    view_mock.assert_not_called()


# Nanslice Overlay test


def test_nanslice_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the expected layers and return the slices result."""
    result = cast(
        "mod.PermutedOLSResult",
        {
            "tested_var_names": ["age", "sex"],
            "beta": np.array(
                [
                    [100.0, 200.0, 300.0],
                    [-5.0, 0.0, 5.0],
                ]
            ),
            "logp_max_t": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [1.0, 2.0, 3.0],
                ]
            ),
        },
    )
    masker = cast(
        "MultiNiftiMasker", Mock(spec="nilearn.maskers.MultiNiftiMasker")
    )

    images: list[Nifti1Image] = [
        Nifti1Image(np.zeros((2, 2, 2)), affine=np.eye(4)) for _ in range(4)
    ]
    bg_img, beta_img, logp_img, alpha_img = images

    base_layer: Mock = Mock()
    beta_layer: Mock = Mock()
    expected_plot: Mock = Mock()

    unmask_mock: Mock = Mock(side_effect=[beta_img, logp_img])
    math_mock: Mock = Mock(return_value=alpha_img)
    layer_mock: Mock = Mock(side_effect=[base_layer, beta_layer])
    slices_mock: Mock = Mock(return_value=expected_plot)

    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "math_img", math_mock)
    monkeypatch.setattr(mod, "Layer", layer_mock)
    monkeypatch.setattr(mod, "slices", slices_mock)

    actual_plot = mod.nanslice_overlay(
        result,
        masker,
        bg_img,
        term="sex",
        p_thresh=0.01,
        cmap="coolwarm",
        title="Sex effect",
    )

    # Check that both arrays come from the selected term's row.
    expected_unmask_call_count = 2
    assert unmask_mock.call_count == expected_unmask_call_count
    np.testing.assert_array_equal(
        unmask_mock.call_args_list[0].args[0],
        [-5.0, 0.0, 5.0],
    )
    np.testing.assert_array_equal(
        unmask_mock.call_args_list[1].args[0],
        [1.0, 2.0, 3.0],
    )
    for recorded_call in unmask_mock.call_args_list:
        assert recorded_call.args[1] is masker

    math_mock.assert_called_once_with(
        "1 - np.power(10, -img)",
        img=logp_img,
    )

    expected_layer_call_count = 2
    assert layer_mock.call_count == expected_layer_call_count
    assert layer_mock.call_args_list[0] == call(
        bg_img,
        cmap="gray",
        label="Brain template",
    )

    overlay_call = layer_mock.call_args_list[1]
    assert overlay_call.args == (beta_img,)
    assert overlay_call.kwargs["cmap"] == "coolwarm"

    # The 2nd and 98th percentiles of [-5, 0, 5] are -4.8 and 4.8.
    np.testing.assert_allclose(
        overlay_call.kwargs["clim"],
        [-4.8, 4.8],
    )
    assert overlay_call.kwargs["alpha"] is alpha_img
    assert overlay_call.kwargs["alpha_lim"] == (0, 1)
    assert overlay_call.kwargs["alpha_label"] == "1 - p"
    assert overlay_call.kwargs["label"] == "Regression coefficient"

    slices_mock.assert_called_once_with(
        [base_layer, beta_layer],
        nrows=5,
        ncols=5,
        cbar=1,
        contour=0.99,
        title="Sex effect",
    )
    assert actual_plot is expected_plot


def test_nanslice_overlay_missing_p_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject results without the requested p-value variable."""
    result = cast(
        "mod.PermutedOLSResult",
        {"tested_var_names": ["age"]},
    )
    masker = cast("MultiNiftiMasker", Mock(spec=MultiNiftiMasker))
    bg_img: Nifti1Image = Nifti1Image(
        np.zeros((2, 2, 2)),
        affine=np.eye(4),
    )
    unmask_mock: Mock = Mock()
    slices_mock: Mock = Mock()
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "slices", slices_mock)

    with pytest.raises(
        ValueError,
        match="Specified p-value variable logp_max_t is not in results",
    ):
        mod.nanslice_overlay(result, masker, bg_img, term="age")

    unmask_mock.assert_not_called()
    slices_mock.assert_not_called()


# Unit test for save_stat_map


def test_save_stat_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unmask the selected statistic row and save the resulting image."""
    filename: Path = tmp_path / "stat_map.nii.gz"
    result: dict[str, np.typing.NDArray[Any]] = {
        "t": np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    }
    masker = cast("MultiNiftiMasker", Mock(spec=MultiNiftiMasker))
    image: Nifti1Image = Nifti1Image(
        np.zeros((3, 3, 3)),
        affine=np.eye(4),
    )

    unmask_mock: Mock = Mock(return_value=image)
    save_mock: Mock = Mock()
    monkeypatch.setattr(mod, "unmask", unmask_mock)
    monkeypatch.setattr(mod, "nib_save", save_mock)

    mod.save_stat_map(filename, result, masker, idx=1, stat="t")

    unmask_mock.assert_called_once()
    np.testing.assert_array_equal(
        unmask_mock.call_args.args[0],
        [4.0, 5.0, 6.0],
    )
    assert unmask_mock.call_args.args[1] is masker
    save_mock.assert_called_once_with(image, filename)


# Test for save_all_maps function


def test_save_all_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create the output directory and save available statistics."""
    outputdir: Path = tmp_path / "results" / "maps"
    masker = cast("MultiNiftiMasker", Mock(spec=MultiNiftiMasker))

    result: dict[str, np.typing.NDArray[Any]] = {
        "tested_var_names": np.array(["age (years)", "group[T.patient]"]),
        "t": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "logp_max_t": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    # Include beta in the options, but leave it out of result.
    monkeypatch.setattr(mod, "STAT_OPTIONS", ["t", "beta"])
    monkeypatch.setattr(mod, "P_OPTIONS", ["logp_max_t"])

    save_mock: Mock = Mock()
    monkeypatch.setattr(mod, "save_stat_map", save_mock)

    filenames: list[str] = mod.save_all_maps(
        outputdir,
        result,
        masker,
        imgvar="brain",
    )

    expected_names: list[str] = [
        "brain_contrast-ageyears_stat-t_map.nii.gz",
        "brain_contrast-groupTpatient_stat-t_map.nii.gz",
        "brain_contrast-ageyears_stat-logpmaxt_map.nii.gz",
        "brain_contrast-groupTpatient_stat-logpmaxt_map.nii.gz",
    ]

    assert outputdir.is_dir()
    assert filenames == expected_names
    assert save_mock.call_args_list == [
        call(outputdir / expected_names[0], result, masker, 0, "t"),
        call(outputdir / expected_names[1], result, masker, 1, "t"),
        call(outputdir / expected_names[2], result, masker, 0, "logp_max_t"),
        call(outputdir / expected_names[3], result, masker, 1, "logp_max_t"),
    ]


def test_save_all_maps_without_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return no filenames when no supported statistics are present."""
    outputdir: Path = tmp_path / "maps"
    masker = cast("MultiNiftiMasker", Mock(spec=MultiNiftiMasker))
    result: dict[str, np.typing.NDArray[Any]] = {
        "tested_var_names": np.array(["age"]),
    }

    monkeypatch.setattr(mod, "STAT_OPTIONS", ["t"])
    monkeypatch.setattr(mod, "P_OPTIONS", ["logp_max_t"])

    save_mock: Mock = Mock()
    monkeypatch.setattr(mod, "save_stat_map", save_mock)

    filenames: list[str] = mod.save_all_maps(
        outputdir,
        result,
        masker,
        imgvar="brain",
    )

    assert filenames == []
    assert outputdir.is_dir()
    save_mock.assert_not_called()
