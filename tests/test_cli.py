"""Test command line interface."""

from click.testing import CliRunner

from neurovoxel.cli import main


def test_cli_help() -> None:
    """Test if CLI help runs."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
