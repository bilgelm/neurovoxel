"""Define the package version."""

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import (
    distributions,
    version,
)
from pathlib import Path
from typing import Any

VERSION = version("Neurovoxel")
timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()  # noqa: S607
packages = {dist.metadata["Name"]: dist.version for dist in distributions()}


def save_parameters(
    filename: Path,
    data: dict[str, Any],
) -> None:
    """Save parameters to a JSON file."""
    with filename.open("w") as f:
        json.dump(data, f, indent=4)


def parameter_output(  # noqa: PLR0913
    paths: dict[str, str],
    analysis: dict[str, str | float | int | bool | None],
    statistical_maps: list[str],
    tabular_summary: str,
    atlasreader_output: list[str] | None,
    cluster_size: int | None,
) -> dict[
    str,
    dict[str, str]
    | dict[str, str | float | int | bool | None]
    | str
    | list[str]
    | Any
    | None,
]:
    """Return a dictionary of parameters for saving."""
    return {
        "paths": paths,
        "analysis": analysis,
        "version": VERSION,
        "timestamp (UTC)": timestamp,
        "commit": commit,
        "statistical maps in output folder": statistical_maps,
        "tabular summary in output folder": tabular_summary,
        "atlasreader output in output folder": atlasreader_output,
        "packages": packages,
        "atlasreadercluster size": cluster_size,
    }
