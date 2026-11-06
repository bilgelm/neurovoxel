"""Unit tests for the footer component."""

from types import SimpleNamespace

import pytest

import neurovoxel.components.footer as mod


def test_render_footer_passes_html_to_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The footer should render its text as raw HTML via st.markdown."""
    calls = SimpleNamespace(args=None, kwargs=None)

    def fake_markdown(*args: object, **kwargs: object) -> None:
        calls.args, calls.kwargs = args, kwargs

    monkeypatch.setattr(mod.st, "markdown", fake_markdown)

    mod.render_footer()

    assert "NeuroVoxel" in calls.args[0]
    assert "<hr>" in calls.args[0]
    assert calls.kwargs["unsafe_allow_html"] is True
