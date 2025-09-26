"""Features subpackage for NeuroVoxel app."""

import json
from pathlib import Path

with (Path(__file__).parent / "template.json").open("r") as f:
    SCHEMA = json.load(f)

ZERO_VOXEL_OPTS = SCHEMA["properties"]["analysis"]["properties"][
    "handle_zero_voxels"
]["enum"]
MULTI_SES_OPTS = SCHEMA["properties"]["analysis"]["properties"][
    "handle_multiple_sessions"
]["enum"]
STANDARDIZATION_OPTS = SCHEMA["properties"]["analysis"]["properties"][
    "voxelwise_standardization"
]["enum"]
