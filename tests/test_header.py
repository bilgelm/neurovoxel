"""Unit tests for the header component."""

import pytest

import neurovoxel.components.header as mod


def test_render_header_renders_logo_and_tagline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header should show the logo and the tagline as raw HTML."""
    markdown_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    image_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_markdown(*args: object, **kwargs: object) -> None:
        markdown_calls.append((args, kwargs))

    def fake_image(*args: object, **kwargs: object) -> None:
        image_calls.append((args, kwargs))

    monkeypatch.setattr(mod.st, "markdown", fake_markdown)
    monkeypatch.setattr(mod.st, "image", fake_image)

    mod.render_header()

    # Logo: one call, with the expected path and width
    assert len(image_calls) == 1
    assert image_calls[0][0] == ("assets/logo.png",)
    assert image_calls[0][1]["width"] == 500  # noqa: PLR2004

    # Three markdown calls: wrapper open, tagline, wrapper close
    assert len(markdown_calls) == 3  # noqa: PLR2004
    assert all(
        kwargs["unsafe_allow_html"] is True for _, kwargs in markdown_calls
    )

    html = [args[0] for args, _ in markdown_calls]
    assert all(isinstance(content, str) for content in html)
    assert "flex-direction:column" in str(html[0])
    assert "Statistical Analysis Platform for Neuroimaging Datasets" in str(
        html[1]
    )
    assert html[2] == "</div>"
