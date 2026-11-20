"""Model fit maps, PDF reports and mixed-model report integration."""

# pyright: reportMissingTypeStubs=false

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from nibabel.loadsave import load  # pyright: ignore[reportUnknownVariableType]
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import MultiNiftiMasker

import neurovoxel.utils.model_diagnostics as reports
from neurovoxel import app
from neurovoxel.utils.viz import save_all_maps


@pytest.fixture
def masker() -> MultiNiftiMasker:
    """Fit a mask with both included and excluded spatial locations."""
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 0
    return MultiNiftiMasker(
        mask_img=Nifti1Image(mask, np.eye(4)), reports=False
    ).fit()  # pyright: ignore[reportUnknownMemberType]


def test_model_fit_maps_saved_once_per_model(
    tmp_path: Path, masker: MultiNiftiMasker
) -> None:
    """Save one 3D map per diagnostic, preserving failed and outside NaNs."""
    values = np.array([np.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = {
        "tested_var_names": np.array(["age", "sex"]),
        **{
            key: values.copy()
            for key in (
                "r_square",
                "r_square_conditional",
                "f_stat",
                "f_pvalues",
            )
        },
    }
    names = save_all_maps(tmp_path, result, masker, "gm")
    assert names == [
        "gm_stat-rsquare_map.nii.gz",
        "gm_stat-rsquareconditional_map.nii.gz",
        "gm_stat-fstat_map.nii.gz",
        "gm_stat-fpvalues_map.nii.gz",
    ]
    for name in names:
        saved = cast("Nifti1Image", load(tmp_path / name)).get_fdata()
        assert saved.shape == (2, 2, 2)
        assert np.isnan(saved[0, 0, 0])
        np.testing.assert_allclose(saved.ravel()[1:], values)


@pytest.mark.parametrize("mixed", [False, True])
def test_report_uses_available_statistics_and_model_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mixed: bool
) -> None:
    """Include conditional R² for LME and skip unavailable overall tests."""
    result: dict[str, Any] = {
        "r_square": np.array([0.2, np.nan]),
        "f_stat": np.full(2, np.nan),
        "f_pvalues": np.full(2, np.nan),
    }
    if mixed:
        result.update(
            r_square_conditional=np.array([0.8, np.nan]), mixed_diagnostics={}
        )
    state = SimpleNamespace(
        result=result, paths={"outputdir": str(tmp_path)}, masker=object()
    )
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=state))
    calls: list[dict[str, Any]] = []

    def save(**kwargs: Any) -> None:  # noqa: ANN401
        calls.append(kwargs)

    monkeypatch.setattr(app, "generate_ols_results_pdf", save)
    app._save_model_fit_report()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    assert len(calls) == 1
    assert calls[0]["output_path"] == tmp_path / (
        "lme_results.pdf" if mixed else "ols_results.pdf"
    )
    assert calls[0]["mixed"] is mixed
    assert set(calls[0]["stat_values"]) == (
        {"r_square", "r_square_conditional"} if mixed else {"r_square"}
    )


def test_mixed_pdf_labels_and_nan_histograms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    masker: MultiNiftiMasker,
) -> None:
    """Write a PDF with explicit marginal/conditional and asymptotic labels."""
    titles: list[str] = []

    def plot(*_args: object, **kwargs: Any) -> None:  # noqa: ANN401
        titles.append(kwargs["title"])

    monkeypatch.setattr(reports.plotting, "plot_stat_map", plot)
    values: dict[str, reports.StatValues] = {
        key: {"array": np.array([0.1] * 6 + [np.nan]), "vmax": 1.0}
        for key in (
            "r_square",
            "r_square_conditional",
            "f_stat",
            "f_pvalues",
        )
    }
    output = tmp_path / "lme_results.pdf"
    reports.generate_ols_results_pdf(
        values, masker, Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), output
    )
    assert output.read_bytes().startswith(b"%PDF")
    assert any("Marginal" in title for title in titles)
    assert any("Conditional" in title for title in titles)
    assert any("Asymptotic" in title for title in titles)
    assert any("uncorrected" in title for title in titles)
