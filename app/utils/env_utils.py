"""Utility functions for environment configuration handling.

This module provides functions to determine if the current environment is set for testing
by reading configuration from a YAML file.
"""

from pathlib import Path
from typing import Optional

import yaml

def is_testing_env(config_path: Optional[str | Path] = None) -> bool:
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"
    else:
        config_path = Path(config_path)
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        return bool(config and "environment" in config and config["environment"].get("testing", False))
    except Exception:
        return False
