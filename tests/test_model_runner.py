"""Testing the model runner component."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, sentinel

import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import neurovoxel.components.model_runner as mod


class FakeSessionState(dict[str, object]):
    """Support both dictionary and attribute access."""

    def __getattr__(self, name: str) -> object:
        """Return a stored value for attribute-style access."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        """Store a value using attribute-style access."""
        self[name] = value


@pytest.mark.parametrize(
    ("overrides", "expected_n_jobs", "expected_random_state"),
    [
        ({}, -1, 42),
        ({"n_jobs": 2, "random_state": 123, "correction": "fdr"}, 2, 123),
        ({"query": "gm ~ age + (age | subject)", "correction": "fdr"}, -1, 42),
    ],
    ids=["default-options", "custom-options", "mixed-preparation-progress"],
)
def test_render_model_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object],
    expected_n_jobs: int,
    expected_random_state: int,
) -> None:
    """Verify model runner configuration and saved outputs."""
    input_tbl = pd.DataFrame({"subject": ["01"]})
    output_tbl = pd.DataFrame({"subject": ["01"], "included": [True]})
    images = [sentinel.image]

    analysis = {
        "query": "gm ~ age",
        "inference_terms": ["age"],
        "smoothing_fwhm": 6,
        "voxel_size": 2,
        "n_perm": 100,
        "tfce": True,
        "handle_zero_voxels": "error",
        "handle_multiple_sessions": "all",
        **overrides,
    }

    layout = Mock()
    layout.get.return_value = images

    state = FakeSessionState(
        entity_df=pd.DataFrame(
            [
                {
                    "name": "wm",
                    "suffix": "probseg",
                    "label": "WM",
                    "task": "rest",
                },
                {
                    "name": "gm",
                    "suffix": "probseg",
                    "label": "GM",
                    "task": None,
                },
            ]
        ),
        layout=layout,
        paths={
            "mask": str(tmp_path / "mask.nii.gz"),
            "bids_root": str(tmp_path),
        },
        analysis=analysis,
        tbl=input_tbl,
    )

    spinner = MagicMock()
    fake_st = SimpleNamespace(
        session_state=state, spinner=spinner, progress=Mock()
    )

    mixed = "|" in str(analysis["query"])

    def prepare_masker(*_args: object, **_kwargs: object) -> object:
        # The progress widget must exist before preprocessing starts, even
        # when no voxel-fitting callback has fired yet.
        if mixed:
            fake_st.progress.assert_called_once()
        return sentinel.masker

    get_masker = Mock(side_effect=prepare_masker)
    run_query = Mock(return_value=(sentinel.result, output_tbl))

    monkeypatch.setattr(mod, "st", fake_st)
    monkeypatch.setattr(mod, "get_masker", get_masker)
    monkeypatch.setattr(mod, "run_query", run_query)

    returned = mod.render_model_runner("gm")

    # Select the requested entity and remove name/missing columns.
    layout.get.assert_called_once_with(suffix="probseg", label="GM")

    get_masker.assert_called_once_with(
        str(tmp_path / "mask.nii.gz"),
        6,
        2,
        n_jobs=expected_n_jobs,
    )

    run_query.assert_called_once()
    args, kwargs = run_query.call_args

    assert args[0] == analysis["query"]
    assert args[1] == tmp_path
    assert args[2] == ["age"]
    assert args[3] is input_tbl
    assert args[4] is images
    assert args[5] is sentinel.masker

    assert callable(kwargs.pop("progress_callback"))
    assert callable(kwargs.pop("diagnostics_callback"))
    if mixed:
        fake_st.progress.assert_called_once_with(
            0.0,
            text="Preparing LME data: matching images, loading and "
            "preprocessing...",
        )
        fake_st.progress.return_value.empty.assert_called_once_with()
    else:
        fake_st.progress.assert_not_called()
    assert kwargs == {
        "correction": overrides.get("correction", "fwe"),
        "voxelwise_standardization": mod.STANDARDIZATION_OPTS[0],
        "n_perm": 100,
        "n_jobs": expected_n_jobs,
        "random_state": expected_random_state,
        "tfce": True,
        "handle_zero_voxels": "error",
        "handle_multiple_sessions": "all",
    }

    # Save the analysis outputs into session state.
    assert state.masker is sentinel.masker
    assert state.result is sentinel.result
    assert state.tbl is output_tbl
    assert returned is None

    spinner.assert_called_once_with("Running analysis...")
    spinner.return_value.__enter__.assert_called_once()
    spinner.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_lme_progress_updates_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Throttle intermediate updates, always finish, and clear on exceptions."""
    fake_st = Mock()
    monkeypatch.setattr(mod, "st", fake_st)
    monkeypatch.setattr(mod, "monotonic", lambda: 10.0)

    def run() -> None:
        with mod._lme_progress() as update:  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            update(0, 2)
            update(1, 2)
            if fail:
                msg = "Fit interrupted"
                raise RuntimeError(msg)
            update(2, 2)

    if fail:
        with pytest.raises(RuntimeError, match="Fit interrupted"):
            run()
        fake_st.progress.return_value.progress.assert_not_called()
    else:
        run()
        fake_st.progress.return_value.progress.assert_called_once_with(
            1.0, text="Fitting LME: 2/2 voxels (100%)"
        )
    fake_st.progress.assert_called_once_with(
        0.0, text="Fitting LME: 0/2 voxels (0%)"
    )
    fake_st.progress.return_value.empty.assert_called_once_with()


def test_preparation_progress_switches_immediately_to_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not throttle away the first fitting update after preparation."""
    fake_st = Mock()
    monkeypatch.setattr(mod, "st", fake_st)
    monkeypatch.setattr(mod, "monotonic", lambda: 0.0)
    with mod._lme_progress(prepare=True) as update:  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        fake_st.progress.assert_called_once()
        update(0, 10)
        fake_st.progress.return_value.progress.assert_called_once_with(
            0.0, text="Fitting LME: 0/10 voxels (0%)"
        )
    fake_st.progress.return_value.empty.assert_called_once_with()


@pytest.mark.parametrize("fail", [False, True])
def test_runner_saves_diagnostics_before_return_or_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail: bool,
) -> None:
    """Persist runner diagnostics on successful and failed runs."""
    from neurovoxel.utils.mixed import MixedModelFitError  # noqa: PLC0415
    from neurovoxel.utils.mixed_diagnostics import (  # noqa: PLC0415
        DIAGNOSTICS_FILENAME,
        SUMMARY_FILENAME,
        MixedDiagnostics,
    )

    diagnostics: MixedDiagnostics = {
        "summary": {
            "n_failed": int(fail),
            "final_failure_reasons": {"nonconverged": 1} if fail else {},
        },
        "voxels": [
            {
                "voxel_index": 0,
                "succeeded": not fail,
                "successful_optimizer": None,
                "attempts": [],
            }
        ],
    }
    table = pd.DataFrame({"subject": ["a"]})
    image = Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))

    def get_images(**_kwargs: object) -> list[object]:
        return []

    def get_masker(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(mask_img_=image)

    def run_query(
        *_args: object, **kwargs: object
    ) -> tuple[dict[str, object], pd.DataFrame]:
        callback = kwargs["diagnostics_callback"]
        assert callable(callback)
        callback(diagnostics)
        if fail:
            raise MixedModelFitError(diagnostics)
        return {"mixed_diagnostics": diagnostics}, table

    def spinner(_label: str) -> nullcontext[None]:
        return nullcontext()

    def progress(*_args: object, **_kwargs: object) -> SimpleNamespace:
        def empty() -> None:
            pass

        return SimpleNamespace(empty=empty)

    state = FakeSessionState(
        entity_df=pd.DataFrame({"name": ["gm"], "suffix": ["probseg"]}),
        layout=SimpleNamespace(get=get_images),
        paths={"bids_root": str(tmp_path)},
        analysis={"query": "gm ~ age + (1|subject)"},
        tbl=table,
    )
    monkeypatch.setattr(
        mod,
        "st",
        SimpleNamespace(
            session_state=state, spinner=spinner, progress=progress
        ),
    )
    monkeypatch.setattr(mod, "get_masker", get_masker)
    monkeypatch.setattr(mod, "run_query", run_query)
    if fail:
        with pytest.raises(MixedModelFitError):
            mod.render_model_runner("gm", diagnostics_outputdir=tmp_path)
    else:
        mod.render_model_runner("gm", diagnostics_outputdir=tmp_path)
    assert state["mixed_diagnostics"] is diagnostics
    assert json.loads((tmp_path / SUMMARY_FILENAME).read_text())[
        "n_failed"
    ] == int(fail)
    record = json.loads((tmp_path / DIAGNOSTICS_FILENAME).read_text())
    assert record["succeeded"] == (not fail)
