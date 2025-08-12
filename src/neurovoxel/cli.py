"""NeuroVoxel commands."""

import subprocess
import sys

import click


def run_app() -> None:
    """Start the NeuroVoxel streamlit app."""
    streamlit_app_path = "src/neurovoxel/app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        streamlit_app_path,
        "--server.fileWatcherType",
        "none",
    ]
    subprocess.run(cmd, check=False)  # noqa: S603


@click.command()
def main() -> None:
    """NeuroVoxel command line interface."""


if __name__ == "__main__":
    main()
