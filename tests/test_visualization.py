"""Test of component visualization."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

import neurovoxel.components.visualization as mod


@pytest.mark.parametrize("template", ["/data/template.nii.gz", None])
def test_render_visualization(
    monkeypatch: pytest.MonkeyPatch, template: str | None
) -> None:
    """Test rendering visualization with and without a template."""
    result = object()
    masker = object()
    paths = {"template": template} if template else {}

    def get_state(key: str, default: object = None) -> object:
        return paths if key == "paths" else default

    fake_st = MagicMock()
    fake_st.session_state = SimpleNamespace(
        result=result,
        masker=masker,
        get=get_state,
    )

    views = [MagicMock(), MagicMock()]
    views[0].get_iframe.return_value = "<iframe>age</iframe>"
    views[1].get_iframe.return_value = "<iframe>sex</iframe>"

    figures = [object(), object()]
    interactive_viz = MagicMock(side_effect=views)
    overlay = MagicMock(side_effect=figures)

    monkeypatch.setattr(mod, "st", fake_st)
    monkeypatch.setattr(mod, "basic_interactive_viz", interactive_viz)
    monkeypatch.setattr(mod, "nanslice_overlay", overlay)

    terms = ["age", "sex"]
    mod.render_visualization(pd.Index(terms))

    assert interactive_viz.call_args_list == [
        call(
            result,
            masker,
            term=term,
            stat="t",
            bg_img=template,
            opacity=0.5,
            title=f"t-stat for {term}",
        )
        for term in terms
    ]

    for view in views:
        view.get_iframe.assert_called_once_with()

    assert fake_st.components.v1.html.call_args_list == [
        call("<iframe>age</iframe>", width=650, height=250),
        call("<iframe>sex</iframe>", width=650, height=250),
    ]

    assert overlay.call_args_list == [
        call(
            result,
            masker,
            bg_img=template,
            term=term,
            p_var="logp_max_t",
            p_thresh=0.01,
            title=(
                "Regression coefficient overlay with "
                f"corrected p < 0.01 contours for {term}"
            ),
        )
        for term in terms
    ]

    assert fake_st.pyplot.call_args_list == [
        call(figure, width=1300) for figure in figures
    ]


def test_render_visualization_no_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rendering visualization when there are no terms."""
    fake_st = MagicMock()
    interactive_viz = MagicMock()
    overlay = MagicMock()

    monkeypatch.setattr(mod, "st", fake_st)
    monkeypatch.setattr(mod, "basic_interactive_viz", interactive_viz)
    monkeypatch.setattr(mod, "nanslice_overlay", overlay)

    mod.render_visualization(pd.Index([]))

    interactive_viz.assert_not_called()
    overlay.assert_not_called()
    fake_st.components.v1.html.assert_not_called()
    fake_st.pyplot.assert_not_called()
