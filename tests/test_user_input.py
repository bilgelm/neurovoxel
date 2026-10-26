"""Unit tests for render_bids_input: the BIDS root directory."""

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest

import neurovoxel.components.user_input as mod

#Unit test for test_bids_root

class FakeSessionState(dict[str, object]):
    """Dictionary-backed session state exposing the paths mapping."""

    @property
    def paths(self) -> object:
        """Return the paths mapping from session state."""
        return self["paths"]


@pytest.fixture
def mocked_st(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide a mocked Streamlit interface with isolated session state."""
    fake_st = Mock()
    fake_st.session_state = FakeSessionState(paths={})
    monkeypatch.setattr(mod, "st", fake_st)
    return fake_st


def test_bids_root_is_directory(
        mocked_st: Mock, tmp_path: Path
    ) -> None:
    """Verify an existing directory is accepted as the BIDS root."""
    # tmp_path is an actual directory created by pytest.
    bids_root = str(tmp_path)
    mocked_st.empty.return_value.text_input.return_value = bids_root

    result = mod.render_bids_input()

    assert result is True
    assert mocked_st.session_state.paths["bids_root"] == bids_root
    mocked_st.error.assert_not_called()
    mocked_st.write.assert_not_called()


def test_bids_root_does_not_exist(
    mocked_st: Mock, tmp_path: Path
) -> None:
    """Verify a nonexistent path is rejected as the BIDS root."""
    # This path is not created.
    bids_root = str(tmp_path / "missing_directory")
    mocked_st.empty.return_value.text_input.return_value = bids_root

    result = mod.render_bids_input()

    assert result is False
    mocked_st.error.assert_called_once_with(
        f"Directory does not exist: {bids_root}"
    )


def test_bids_root_not_given(mocked_st: Mock) -> None:
    """Verify clearing the BIDS root removes the stored value."""
    # Start with an old value to check that clearing the input removes it.
    mocked_st.session_state.paths["bids_root"] = "/old/directory"
    mocked_st.empty.return_value.text_input.return_value = ""

    result = mod.render_bids_input()

    assert result is False
    assert "bids_root" not in mocked_st.session_state.paths
    mocked_st.write.assert_called_once_with(
        "❗️ No BIDS root directory selected."
    )
    mocked_st.error.assert_not_called()




# test for render table input

@pytest.mark.parametrize("var_name", ["template", "mask"])
def test_template_input_existing_file(
    mocked_st: Mock,
    tmp_path: Path,
    var_name: Literal["template", "mask"],
) -> None:
    """Verify an existing image file is accepted as a template or mask."""
    image_path = tmp_path / "image.nii.gz"
    image_path.touch()
    mocked_st.empty.return_value.text_input.return_value = str(image_path)

    result = mod.render_template_input(
        label="Choose image",
        var_name=var_name,
    )

    assert result is True
    assert mocked_st.session_state.paths[var_name] == str(image_path)
    mocked_st.error.assert_not_called()
    mocked_st.write.assert_not_called()


@pytest.mark.parametrize("var_name", ["template", "mask"])
def test_template_input_missing_file(
    mocked_st: Mock,
    tmp_path: Path,
    var_name: Literal["template", "mask"],
) -> None:
    """Verify a nonexistent image file is rejected for template inputs."""
    image_path = tmp_path / "missing.nii.gz"
    mocked_st.empty.return_value.text_input.return_value = str(image_path)

    result = mod.render_template_input(
        label="Choose image",
        var_name=var_name,
    )

    assert result is False
    mocked_st.error.assert_called_once_with(
        f"File does not exist: {image_path}"
    )


@pytest.mark.parametrize("var_name", ["template", "mask"])
def test_template_input_not_given(
    mocked_st: Mock, var_name: Literal["template", "mask"]
) -> None:
    """Verify clearing template or mask input removes the stored value."""
    mocked_st.empty.return_value.text_input.return_value = ""

    result = mod.render_template_input(
        label="Choose image",
        var_name=var_name,
    )

    assert result is False
    mocked_st.write.assert_called_once_with(
        f"❗️ No brain {var_name} image selected."
    )
    mocked_st.error.assert_not_called()


@pytest.mark.parametrize("autoload", [False, True])
def test_template_input_autoload(
    mocked_st: Mock, tmp_path: Path, autoload: bool
) -> None:
    """Verify autoload behavior for template image inputs."""
    image_path = tmp_path / "image.nii.gz"
    image_path.touch()
    input_box = mocked_st.empty.return_value
    input_box.text_input.return_value = str(image_path)

    result = mod.render_template_input(
        label="Choose image",
        var_name="template",
        autoload=autoload,
    )

    assert result is True
    if autoload:
        input_box.empty.assert_called_once_with()
    else:
        input_box.empty.assert_not_called()



#test for render table input

@pytest.mark.parametrize(
    ("extension", "separator"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_render_table_input_valid(
    mocked_st: Mock,
    tmp_path: Path,
    extension: str,
    separator: str,
) -> None:
    """Verify valid CSV and TSV files are loaded as table inputs."""
    # Create a real temporary table file.
    table_path = tmp_path / f"data{extension}"
    expected_table: pd.DataFrame = pd.DataFrame({
        "subject": ["01", "02"],
        "session": ["1", "1"],
        "age": [25, 30],
    })
    expected_table.to_csv(table_path, sep=separator, index=False)

    # Simulate the user entering its path.
    mocked_st.empty.return_value.text_input.return_value = str(table_path)

    result = mod.render_table_input()

    assert result is True
    assert mocked_st.session_state.paths["tabular"] == str(table_path)
    pd.testing.assert_frame_equal(
        mocked_st.session_state.tbl,
        expected_table,
    )
    mocked_st.error.assert_not_called()

def test_table_input_shows_error_for_missing_file(
    mocked_st: Mock, tmp_path: Path
) -> None:
    """Verify that a missing file path results in an error message."""
    missing_path = tmp_path / "missing.csv"
    mocked_st.empty.return_value.text_input.return_value = str(missing_path)

    result = mod.render_table_input()

    assert result is False
    mocked_st.error.assert_called_once_with(
        f"File does not exist: {missing_path}"
    )
    mocked_st.write.assert_not_called()

def test_table_input_shows_error_for_missing_columns(
    mocked_st: Mock, tmp_path: Path
) -> None:
    """Verify that a table file missing.

    required columns results in an error message.
    """
    table_path = tmp_path / "data.csv"
    table_path.write_text("age\n25\n", encoding="utf-8")
    mocked_st.empty.return_value.text_input.return_value = str(table_path)

    result = mod.render_table_input()

    assert result is False
    mocked_st.error.assert_called_once_with(
        "Tabular data file must contain 'subject' and 'session' columns."
    )
    mocked_st.write.assert_not_called()

def test_table_input_shows_message_when_not_selected(mocked_st: Mock) -> None:
    """Verify clearing the table input.

    shows a message and removes stored values.
    """
    mocked_st.session_state.paths.pop("tabular", None)
    mocked_st.empty.return_value.text_input.return_value = ""

    result = mod.render_table_input()

    assert result is False
    mocked_st.write.assert_called_once_with(
        "❗️ No tabular file selected."
    )
    mocked_st.error.assert_not_called()



# Tests for render_analysis_param_input

class AnalysisSessionState(dict[str, object]):
    """Provide attribute-style access to analysis session state."""

    @property
    def paths(self) -> object:
        """Return the paths mapping from analysis session state."""
        return self["paths"]

    @property
    def analysis(self) -> object:
        """Return the analysis mapping from analysis session state."""
        return self["analysis"]


@pytest.fixture
def analysis_st(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Configure and return the mocked.

    Streamlit interface for analysis tests.
    """
    state = AnalysisSessionState(
        paths={},
        analysis={"handle_multiple_sessions": "first"},
    )

    fake_st = MagicMock()
    fake_st.session_state = state
    fake_st.text_input.return_value = ""

    # Return each widget's initial value unless a test overrides it.
    def number_input(_label: str, **kwargs: object) -> object:
        value = kwargs.get("value")
        if value is not None:
            return value
        key = str(kwargs.get("key", ""))
        return state.get(key, 0)

    fake_st.number_input.side_effect = number_input
    def checkbox(**kwargs: object) -> bool:
        return bool(kwargs.get("value", False))

    fake_st.checkbox.side_effect = checkbox
    def selectbox(
        _label: str, options: list[object], **kwargs: object
    ) -> object:
        index = kwargs.get("index", 0)
        if not isinstance(index, int):
            index = 0
        return options[index]

    fake_st.selectbox.side_effect = selectbox
    fake_st.columns.return_value = (MagicMock(), MagicMock())

    monkeypatch.setattr(mod, "st", fake_st)
    return fake_st

def test_analysis_input_saves_initial_values(analysis_st: MagicMock) -> None:
    """Verify initial analysis input values are saved to session state."""
    mod.render_analysis_param_input()

    assert analysis_st.session_state.analysis == {
        "smoothing_fwhm": 5.0,
        "voxel_size": 4.0,
        "n_perm": 10000,
        "tfce": False,
        "handle_zero_voxels": mod.ZERO_VOXEL_OPTS[0],
        "handle_multiple_sessions": "first",
        "voxelwise_standardization": mod.STANDARDIZATION_OPTS[0],
    }
    assert analysis_st.session_state["bids_cache_input"] == ""
    analysis_st.columns.assert_not_called()
    analysis_st.warning.assert_not_called()

def test_analysis_input_saves_changed_values(analysis_st: MagicMock) -> None:
    """Verify changed analysis input values are saved to session state."""
    voxel_size = 2.0
    n_perm = 500
    selected_numbers = {
        "smoothing_fwhm_input": 8.0,
        "voxel_size_input": voxel_size,
        "n_perm_input": n_perm,
    }
    analysis_st.number_input.side_effect = (
        lambda label, **kwargs: selected_numbers[kwargs["key"]] # pyright: ignore[reportUnknownLambdaType]  # noqa: ARG005
    )
    analysis_st.checkbox.side_effect = (
        lambda label, **kwargs: kwargs["key"] == "tfce_input" # pyright: ignore[reportUnknownLambdaType]  # noqa: ARG005
    )

    mod.render_analysis_param_input()

    analysis = analysis_st.session_state.analysis
    assert analysis["smoothing_fwhm"] == selected_numbers[
        "smoothing_fwhm_input"
    ]
    assert analysis["voxel_size"] == voxel_size
    assert analysis["n_perm"] == n_perm
    assert analysis["tfce"] is True

@pytest.mark.parametrize(
    "case",
    ["existing_file", "missing_file", "empty", "directory"],
)
def test_analysis_input_validates_cache(
    analysis_st: MagicMock, tmp_path: Path, case: str
) -> None:
    """Verify cache paths are accepted.

    only when they reference an existing file.
    """
    analysis_st.session_state.paths["bids_cache"] = "/old/cache.parquet"

    if case == "existing_file":
        cache_path = tmp_path / "cache.parquet"
        cache_path.touch()
        entered_value = f"  {cache_path}  "
    elif case == "missing_file":
        cache_path = tmp_path / "missing.parquet"
        entered_value = str(cache_path)
    elif case == "directory":
        cache_path = tmp_path
        entered_value = str(cache_path)
    else:
        entered_value = "   "

    analysis_st.text_input.return_value = entered_value

    mod.render_analysis_param_input()

    if case == "existing_file":
        assert analysis_st.session_state.paths["bids_cache"] == str(cache_path) # pyright: ignore[reportPossiblyUnboundVariable]
        analysis_st.warning.assert_not_called()
    else:
        assert "bids_cache" not in analysis_st.session_state.paths

        if case == "empty":
            analysis_st.warning.assert_not_called()
        else:
            analysis_st.warning.assert_called_once_with(
                "Parquet cache file not found, it will not be used: "
                f"{cache_path}" # pyright: ignore[reportPossiblyUnboundVariable]
            )

@pytest.mark.parametrize("session_option", ["first", "all"])
def test_analysis_input_session_warning(
    analysis_st: MagicMock, session_option: str
) -> None:
    """Verify that selecting all sessions displays an independence warning."""
    analysis_st.session_state.analysis[
        "handle_multiple_sessions"
    ] = session_option

    mod.render_analysis_param_input()

    if session_option == "all":
        analysis_st.warning.assert_called_once_with(
            "Selecting 'all' sessions may violate the independence of "
            "observations assumption for linear regression. "
            "Consider using one session per subject."
        )
    else:
        analysis_st.warning.assert_not_called()

def test_analysis_input_atlasreader_conversions(analysis_st: MagicMock) -> None:
    """Verify cluster-size conversions between voxels and cubic millimeters."""
    state = analysis_st.session_state
    state["space_verification"] = True
    state["cluster_size_voxels"] = 2
    state["cluster_size_mm3"] = 129.0

    mod.render_analysis_param_input()

    analysis_st.columns.assert_called_once_with(2)
    analysis_st.markdown.assert_called_once_with(
        "Minimum cluster size for AtlasReader reporting"
    )
    expected_cluster_size = 2
    assert state["cluster_size"] == expected_cluster_size

    # Find the callbacks passed to the two cluster-size widgets.
    widget_arguments = {
        call.kwargs["key"]: call.kwargs
        for call in analysis_st.number_input.call_args_list
    }

    # Default voxel size is 4 mm: each voxel occupies 4³ = 64 mm³.
    widget_arguments["cluster_size_voxels"]["on_change"]()
    expected_cluster_size_mm3 = 2 * 4**3
    assert state["cluster_size_mm3"] == expected_cluster_size_mm3

    # 129 / 64 is rounded up to 3 voxels.
    state["cluster_size_mm3"] = 129.0
    widget_arguments["cluster_size_mm3"]["on_change"]()
    expected_cluster_size_voxels = 3
    assert state["cluster_size_voxels"] == expected_cluster_size_voxels



# Tests for render_outputdir_input

@pytest.mark.parametrize("autoload", [False, True])
def test_outputdir_existing_empty_directory(
    mocked_st: Mock, tmp_path: Path, autoload: bool
) -> None:
    """Accept an existing empty directory as the output directory."""
    input_box = mocked_st.empty.return_value
    input_box.text_input.return_value = str(tmp_path)

    result = mod.render_outputdir_input(autoload=autoload)

    assert result is True
    assert mocked_st.session_state.paths["outputdir"] == str(tmp_path)
    mocked_st.error.assert_not_called()
    mocked_st.write.assert_not_called()

    if autoload:
        input_box.empty.assert_called_once_with()
    else:
        input_box.empty.assert_not_called()


def test_outputdir_existing_nonempty_directory(
    mocked_st: Mock, tmp_path: Path
) -> None:
    """Reject an existing non-empty directory as the output directory."""
    (tmp_path / "existing.txt").write_text("data", encoding="utf-8")
    input_box = mocked_st.empty.return_value
    input_box.text_input.return_value = str(tmp_path)

    result = mod.render_outputdir_input(autoload=True)

    assert result is False
    mocked_st.error.assert_called_once_with(
        "Please specify an empty directory for output."
    )
    input_box.empty.assert_not_called()


def test_outputdir_creates_missing_directory(
    mocked_st: Mock, tmp_path: Path
) -> None:
    """Create and verify a missing output directory."""
    output_path = tmp_path / "new_parent" / "output"
    mocked_st.empty.return_value.text_input.return_value = str(output_path)

    result = mod.render_outputdir_input()

    assert result is True
    assert output_path.is_dir()
    assert list(output_path.iterdir()) == []
    assert mocked_st.session_state.paths["outputdir"] == str(output_path)
    mocked_st.error.assert_not_called()


def test_outputdir_creation_error(mocked_st: Mock, tmp_path: Path) -> None:
    """Report an error when creating the output directory fails."""
    output_path = tmp_path / "output"
    input_box = mocked_st.empty.return_value
    input_box.text_input.return_value = str(output_path)

    # Simulate failure without changing real filesystem permissions.
    with patch.object(
        Path,
        "mkdir",
        side_effect=OSError("Permission denied"),
    ):
        result = mod.render_outputdir_input(autoload=True)

    assert result is False
    assert not output_path.exists()
    mocked_st.error.assert_called_once_with(
        "Could not create output directory: Permission denied"
    )
    input_box.empty.assert_not_called()


def test_outputdir_not_given(mocked_st: Mock) -> None:
    """Verify clearing the output directory removes the stored value."""
    mocked_st.session_state.paths.pop("outputdir", None)
    mocked_st.empty.return_value.text_input.return_value = ""

    result = mod.render_outputdir_input()

    assert result is False
    mocked_st.write.assert_called_once_with(
        "❗️ No output directory selected."
    )
    mocked_st.error.assert_not_called()




# Unit test for is_directory_empty

def test_is_directory_empty_returns_true(tmp_path: Path) -> None:
    """Verify an empty directory is reported as empty."""
    assert mod.is_directory_empty(tmp_path) is True


def test_is_directory_empty_returns_false_with_file(tmp_path: Path) -> None:
    """Verify a directory containing a file is reported as non-empty."""
    (tmp_path / "file.txt").touch()

    assert mod.is_directory_empty(tmp_path) is False


def test_is_directory_empty_returns_false_with_subdirectory(
        tmp_path: Path
) -> None:
    """Verify a directory containing a subdirectory is reported as non-empty."""
    (tmp_path / "subdirectory").mkdir()

    assert mod.is_directory_empty(tmp_path) is False


@pytest.mark.parametrize("path_type", ["missing", "file"])
def test_is_directory_empty_rejects_invalid_path(
    tmp_path: Path, path_type: str
) -> None:
    """Verify that a missing path or a file raises a ValueError."""
    path = tmp_path / "invalid"

    if path_type == "file":
        path.touch()

    with pytest.raises(
        ValueError,
        match=f"Path '{path}' is not a directory or does not exist\\.",
    ) as exc_info:
        mod.is_directory_empty(path)

    assert str(exc_info.value) == (
        f"Path '{path}' is not a directory or does not exist."
    )



# Unit test for render_inference_choices

def test_inference_choices_defaults(analysis_st: MagicMock) -> None:
    """Verify that inference choices default to all non-intercept terms."""
    rhs = pd.Index(["Intercept", "age", "sex", "group", "score"])
    analysis_st.columns.return_value = [MagicMock() for _ in range(4)]

    # Simulate accepting each checkbox's default value.
    analysis_st.checkbox.side_effect = (
        lambda label, **kwargs: kwargs["value"] # pyright: ignore[reportUnknownLambdaType]  # noqa: ARG005
    )

    mod.render_inference_choices(rhs)

    assert analysis_st.session_state.analysis["inference_terms"] == {
        "age", "sex", "group", "score"
    }
    analysis_st.text.assert_called_once_with(
        "Select model terms for which to perform inference:"
    )
    analysis_st.columns.assert_called_once_with(4)
    assert analysis_st.checkbox.call_args_list == [
        call(
            term,
            value=term != "Intercept",
            key=f"{term}_viz_yes_no",
        )
        for term in rhs
    ]

@pytest.mark.parametrize(
    "selected_terms",
    [
        set[str](),
        {"age"},
        {"Intercept", "sex"},
    ],
)
def test_inference_choices_saves_user_selection(
    analysis_st: MagicMock, selected_terms: set[str]
) -> None:
    """Verify that user-selected inference terms are saved to session state."""
    rhs = pd.Index(["Intercept", "age", "sex"])
    analysis_st.columns.return_value = [MagicMock() for _ in range(4)]
    analysis_st.checkbox.side_effect = (
        lambda label, **kwargs: label in selected_terms # pyright: ignore[reportUnknownLambdaType]  # noqa: ARG005
    )

    # Check that the function replaces a previous selection.
    analysis_st.session_state.analysis["inference_terms"] = {"old_term"}

    mod.render_inference_choices(rhs)

    assert (
        analysis_st.session_state.analysis["inference_terms"]
        == selected_terms
    )


def test_inference_choices_empty_rhs(analysis_st: MagicMock) -> None:
    """Verify that an empty RHS results in no inference terms."""
    analysis_st.columns.return_value = [MagicMock() for _ in range(4)]
    analysis_st.session_state.analysis["inference_terms"] = {"old_term"}

    mod.render_inference_choices(pd.Index([]))

    assert analysis_st.session_state.analysis["inference_terms"] == set()
    analysis_st.checkbox.assert_not_called()




# Tests for parameter_output

@pytest.mark.parametrize(
    "atlasreader_output",
    [None, [], ["atlasreader/report.csv", "atlasreader/figure.png"]],
)
def test_parameter_output(
    atlasreader_output: list[str] | None,
) -> None:
    """Verify that parameter_output returns the expected dictionary."""
    paths = {
        "bids_root": "/data/bids",
        "outputdir": "/data/output",
    }
    analysis = {
        "smoothing_fwhm": 5.0,
        "n_perm": 1000,
        "tfce": False,
        "handle_multiple_sessions": "first",
        "optional_value": None,
    }
    statistical_maps = ["beta.nii.gz", "pvalue.nii.gz"]
    packages = {"numpy": "test-version", "pandas": "test-version"}

    result = mod.parameter_output(
        paths=paths,
        analysis=analysis,
        version="1.0.0",
        timestamp="2026-10-26T12:00:00",
        commit="abc123",
        statistical_maps=statistical_maps,
        tabular_summary="summary.csv",
        atlasreader_output=atlasreader_output,
        packages=packages,
    )

    assert result == {
        "paths": paths,
        "analysis": analysis,
        "version": "1.0.0",
        "timestamp (UTC)": "2026-10-26T12:00:00",
        "commit": "abc123",
        "statistical maps in output folder": statistical_maps,
        "tabular summary in output folder": "summary.csv",
        "atlasreader output in output folder": atlasreader_output,
        "packages": packages,
    }
