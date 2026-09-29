"""Define the package version."""

import subprocess
from datetime import UTC, datetime
from importlib.metadata import (
    distributions,
    version,
)

VERSION = version("Neurovoxel")
timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()  # noqa: S607
packages = {dist.metadata["Name"]: dist.version for dist in distributions()}
