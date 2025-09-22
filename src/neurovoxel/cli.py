"""NeuroVoxel commands."""

import subprocess
import sys

import click


@click.command()
@click.option("--config-file", help="NeuroVoxel configuration file.")
@click.option("--autoload", is_flag=True, help="Autoload paths")
def run_app(config_file: str | None = None, autoload: bool = False) -> None:
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
    # if there are any options specified, need to first insert "--"
    if config_file or autoload:
        cmd = [*cmd, "--"]
        if config_file:
            cmd = [*cmd, "--config-file", config_file]
        if autoload:
            cmd = [*cmd, "--autoload"]
    subprocess.run(cmd, check=False)  # noqa: S603


@click.command()
def main() -> None:
    """NeuroVoxel command line interface."""


if __name__ == "__main__":
    main()
