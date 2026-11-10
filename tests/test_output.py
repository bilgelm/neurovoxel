"""Unit Test for the output module."""

import json
from pathlib import Path

import pytest

import neurovoxel.utils.output as mod


def test_save_parameters(tmp_path: Path) -> None:
    """Verify that parameters are saved to a JSON file."""
    filename = tmp_path / "parameters.json"
    data = {
        "paths": {"outputdir": "/data/output"},
        "n_perm": 1000,
        "tfce": False,
        "terms": ["age", "sex"],
        "optional": None,
    }

    mod.save_parameters(filename, data)

    assert filename.is_file()
    with filename.open() as f:
        saved_data = json.load(f)

    assert saved_data == data


@pytest.mark.parametrize(
    "atlasreader_output",
    [None, [], ["atlasreader/report.csv"]],
)
def test_parameter_output(
    monkeypatch: pytest.MonkeyPatch,
    atlasreader_output: list[str] | None,
) -> None:
    """Verify that parameter output includes the expected metadata."""
    paths = {"outputdir": "/data/output"}
    analysis: dict[str, str | float | int | bool | None] = {
        "n_perm": 1000,
        "tfce": False,
    }
    statistical_maps = ["beta.nii.gz", "pvalue.nii.gz"]
    packages = {"numpy": "test-version"}

    monkeypatch.setattr(mod, "VERSION", "1.0.0")
    monkeypatch.setattr(mod, "timestamp", "2026-10-26T12:00:00")
    monkeypatch.setattr(mod, "commit", "abc123")
    monkeypatch.setattr(mod, "packages", packages)

    result = mod.parameter_output(
        paths=paths,
        analysis=analysis,
        statistical_maps=statistical_maps,
        tabular_summary="summary.csv",
        atlasreader_output=atlasreader_output,
    )

    assert result == {
        "paths": paths,
        "analysis": analysis,
        "version": "1.0.0",
        "timestamp (UTC)": "2026-10-26T12:00:00",
        "commit": "abc123",
        "statistical maps in output folder": statistical_maps,
        "tabular summary in output folder": "summary.csv",
        "atlasreader output in output folder": atlasreader_output,
        "packages": packages,
    }
