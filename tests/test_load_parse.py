"""Tests for loading and parsing BIDS and configuration data.

This module exercises the helpers in neurovoxel.utils.load_parse, covering
configuration loading, BIDS layout loading, and query parsing.
"""

import json
import secrets
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from bids2table.pybids import (  # pyright: ignore[reportMissingTypeStubs]
    BIDSLayout,  # pyright: ignore[reportMissingTypeStubs]
)
from formulaic.errors import (  # pyright: ignore[reportMissingTypeStubs]
    FactorEvaluationError,  # pyright: ignore[reportMissingTypeStubs]
)
from jsonschema import ValidationError
from nibabel.nifti1 import Nifti1Image

import neurovoxel.utils.load_parse as mod

EXPECTED_N_PERM = 5000

# Test for load_config function


@pytest.fixture
def config() -> dict[str, dict[str, object]]:
    """Return a valid sample configuration."""
    return {
        "analysis": {"n_perm": EXPECTED_N_PERM},
        "paths": {
            "bids_root": "/data/bids",
            "bids_config": "bids_config.json",
            "tabular": "tabular",
            "template": "template",
            "mask": "mask",
            "outputdir": "outputdir",
        },
    }


def test_load_config(
    tmp_path: Path, config: dict[str, dict[str, object]]
) -> None:
    """Test loading a valid configuration."""
    config_file: Path = tmp_path / "valid_config.json"
    config_file.write_text(json.dumps(config))
    result = mod.load_config(config_file)
    assert result["analysis"]["n_perm"] == EXPECTED_N_PERM
    assert result["paths"]["bids_root"] == "/data/bids"


def test_load_config_invalid(
    tmp_path: Path, config: dict[str, dict[str, object]]
) -> None:
    """Test loading an invalid configuration."""
    config["analysis"]["n_perm"] = "invalid"  # supposed to be an integer
    config_file = tmp_path / "invalid_config.json"
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValidationError, match="n_perm"):
        mod.load_config(config_file)


# Test for load_bids


@pytest.fixture
def bids(tmp_path: Path) -> SimpleNamespace:
    """Create a local BIDS dataset with two subjects and two sessions."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test dataset", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    image = Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    for subject in ("01", "02"):
        for session in ("01", "02"):
            anat = root / f"sub-{subject}" / f"ses-{session}" / "anat"
            anat.mkdir(parents=True)
            for suffix, extension in (("T1w", ".nii.gz"), ("FLAIR", ".nii")):
                image.to_filename(  # pyright: ignore[reportUnknownMemberType]
                    anat / f"sub-{subject}_ses-{session}_{suffix}{extension}"
                )

    return SimpleNamespace(
        path=root,
        parquet_path=tmp_path / "bids.parquet",
    )


def test_load_bids(bids: SimpleNamespace) -> None:
    """Test loading a BIDS dataset."""
    result = mod.load_bids(bids.path, bids.parquet_path)
    assert isinstance(result, BIDSLayout)
    assert result.get_subjects() == ["01", "02"]
    assert bids.parquet_path.is_file()


# Test parse_layout function


def test_parse_layout(bids: SimpleNamespace) -> None:
    """Test parsing a BIDS layout."""
    layout = mod.load_bids(bids.path, bids.parquet_path)
    result = mod.parse_layout(layout)
    assert isinstance(result, pd.DataFrame)
    assert {"name", "datatype", "suffix"}.issubset(result.columns)
    expected = pd.DataFrame(
        {
            "name": ["FLAIR", "T1w"],
            "datatype": ["anat", "anat"],
            "suffix": ["FLAIR", "T1w"],
        }
    )
    pd.testing.assert_frame_equal(result[expected.columns], expected)
    assert {"sub", "ses", "ext"}.isdisjoint(result.columns)


@pytest.fixture
def tabular() -> pd.DataFrame:
    """Return a valid tabular dataset."""
    df = pd.DataFrame()
    df["subject"] = list(range(101))
    # Generate random ages between 1 and 101
    df["age"] = [secrets.randbelow(101) + 1 for _ in range(101)]
    return df.iloc[1:].reset_index(drop=True)


def test_parse_query(tabular: pd.DataFrame) -> None:
    """Test parsing a query."""
    lhs, x_mat_columns = mod.parse_query(
        "T1w ~ age", ["T1w", "FLAIR", "dwi"], tabular
    )
    assert lhs == "T1w"
    assert x_mat_columns.tolist() == ["Intercept", "age"]


def test_parse_query_invalid_rhs(tabular: pd.DataFrame) -> None:
    """Test parsing a query: invalid case."""
    with pytest.raises(FactorEvaluationError):
        mod.parse_query(
            "T1w ~ invalid_column", ["T1w", "FLAIR", "dwi"], tabular
        )


def test_parse_query_invalid_lhs(tabular: pd.DataFrame) -> None:
    """Test parsing a query: invalid case."""
    with pytest.raises(ValueError, match="is invalid"):
        mod.parse_query("invalid ~ age", ["T1w", "FLAIR", "dwi"], tabular)


def test_parse_query_no_tilda(tabular: pd.DataFrame) -> None:
    """Test parsing a query: invalid case."""
    with pytest.raises(ValueError, match="Invalid formula syntax"):
        mod.parse_query("invalid age", ["T1w", "FLAIR", "dwi"], tabular)
