"""Tests for data.py."""

from types import SimpleNamespace

import pandas as pd
import pytest

import neurovoxel.components.data as mod

# the module containing render_entity_table


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace `st` in the module with a recorder. Returns the recorder."""
    calls = SimpleNamespace(
        written=[],
        errors=[],
        editor_kwargs=None,
        editor_input=None,
        editor_return=None,  # set by a test to simulate user edits
    )

    def data_editor(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        calls.editor_input = df
        calls.editor_kwargs = kwargs
        # By default, the "edited" table is just the input unchanged
        return calls.editor_return if calls.editor_return is not None else df

    fake = SimpleNamespace(
        write=lambda msg: calls.written.append(msg),  # pyright: ignore[reportUnknownLambdaType]
        error=lambda msg: calls.errors.append(msg),  # pyright: ignore[reportUnknownLambdaType]
        data_editor=data_editor,
    )
    monkeypatch.setattr(mod, "st", fake)
    return calls


def test_only_name_column_is_editable(fake_st: SimpleNamespace) -> None:
    """Verify that only the name column is editable."""
    df = pd.DataFrame(
        {"entity": ["sub", "ses"], "name": ["a", "b"], "count": [3, 2]}
    )

    mod.render_entity_table(df)

    assert fake_st.editor_kwargs["disabled"] == ["entity", "count"]
    assert "name" not in fake_st.editor_kwargs["disabled"]


def test_name_column_moved_to_front(fake_st: SimpleNamespace) -> None:
    """Verify that the name column is moved to the front."""
    df = pd.DataFrame(
        {"entity": ["sub", "ses"], "count": [3, 2], "name": ["a", "b"]}
    )

    mod.render_entity_table(df)

    assert fake_st.editor_kwargs["column_order"] == ["name", "entity", "count"]


def test_editor_options_and_header(fake_st: SimpleNamespace) -> None:
    """Verify editor options and the table header."""
    df = pd.DataFrame({"entity": ["sub"], "name": ["a"]})

    mod.render_entity_table(df)

    assert fake_st.written == ["Types of images in dataset:"]
    assert fake_st.editor_kwargs["key"] == "entity_table_editor"
    assert fake_st.editor_kwargs["hide_index"] is True
    assert fake_st.editor_kwargs["width"] == "content"
    pd.testing.assert_frame_equal(fake_st.editor_input, df)


def test_unique_names_show_no_error(fake_st: SimpleNamespace) -> None:
    """Verify that unique names do not trigger validation errors."""
    df = pd.DataFrame({"entity": ["sub", "ses"], "name": ["a", "b"]})

    result = mod.render_entity_table(df)

    assert fake_st.errors == []
    pd.testing.assert_frame_equal(result, df)


def test_duplicate_names_show_error(fake_st: SimpleNamespace) -> None:
    """Verify that duplicate edited names trigger a validation error."""
    df = pd.DataFrame({"entity": ["sub", "ses"], "name": ["a", "b"]})
    # Simulate the user editing both names to the same value
    fake_st.editor_return = pd.DataFrame(
        {"entity": ["sub", "ses"], "name": ["x", "x"]}
    )

    result = mod.render_entity_table(df)

    assert len(fake_st.errors) == 1
    assert "must be unique" in fake_st.errors[0]
    # The edited table is still returned, duplicates and all
    assert result["name"].tolist() == ["x", "x"]


def test_returns_edited_df_not_input(fake_st: SimpleNamespace) -> None:
    """Verify that the edited dataframe is returned instead of the original."""
    df = pd.DataFrame({"entity": ["sub"], "name": ["a"]})
    fake_st.editor_return = pd.DataFrame(
        {"entity": ["sub"], "name": ["renamed"]}
    )

    result = mod.render_entity_table(df)

    assert result["name"].tolist() == ["renamed"]


def test_without_name_column(fake_st: SimpleNamespace) -> None:
    """Verify behavior when the names column is absent from the table."""
    df = pd.DataFrame({"entity": ["sub", "ses"], "count": [3, 2]})

    result = mod.render_entity_table(df)

    # Everything is disabled, order is unchanged, and no error is raised
    assert fake_st.editor_kwargs["disabled"] == ["entity", "count"]
    assert fake_st.editor_kwargs["column_order"] == ["entity", "count"]
    assert fake_st.errors == []
    pd.testing.assert_frame_equal(result, df)
