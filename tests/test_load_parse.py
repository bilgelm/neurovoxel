"""Tests for loading and parsing BIDS and configuration data.

This module exercises the helpers in neurovoxel.utils.load_parse, covering
configuration loading, BIDS layout loading, and query parsing.
"""

import json
import secrets
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import openneuro
import pandas as pd
import pytest
from bids2table.pybids import (  # pyright: ignore[reportMissingTypeStubs]
    BIDSLayout,  # pyright: ignore[reportMissingTypeStubs]
)
from formulaic.errors import (  # pyright: ignore[reportMissingTypeStubs]
    FactorEvaluationError,  # pyright: ignore[reportMissingTypeStubs]
)
from jsonschema import ValidationError

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


@pytest.fixture(scope="module")
def bids() -> Iterator[SimpleNamespace]:
    """Download BIDS test data and delete it after this module's tests."""
    tag = "1.0.0"

    with TemporaryDirectory(prefix="neurovoxel-test-") as directory:
        root = Path(directory) / "ds007401"

        openneuro.download(
            dataset="ds007401",
            tag=tag,
            target_dir=root,
            include=["sub-3"],
        )

        yield SimpleNamespace(
            path=root,
            parquet_path=Path(directory) / "ds007401.parquet",
        )


def test_load_bids(bids: SimpleNamespace) -> None:
    """Test loading a BIDS dataset."""
    result = mod.load_bids(bids.path, bids.parquet_path)
    assert isinstance(result, BIDSLayout)


# Test parse_layout function


def test_parse_layout(bids: SimpleNamespace) -> None:
    """Test parsing a BIDS layout."""
    layout = mod.load_bids(bids.path, bids.parquet_path)
    result = mod.parse_layout(layout)
    assert isinstance(result, pd.DataFrame)
    assert {"name", "datatype", "suffix"}.issubset(result.columns)


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
