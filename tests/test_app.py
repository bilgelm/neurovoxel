"""Test streamlit app."""

from streamlit.testing.v1 import AppTest


def test_app_runs() -> None:
    """Test if streamlit app runs."""
    AppTest.from_file("src/neurovoxel/app.py").run(timeout=30)
