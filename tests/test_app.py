"""Test streamlit app."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, call, sentinel

import pytest
from streamlit.testing.v1 import AppTest

import neurovoxel.app as mod


def test_app_runs() -> None:
    """Test if streamlit app runs."""
    AppTest.from_file("src/neurovoxel/app.py").run(timeout=30)


# Test for _load_dataset function


@pytest.mark.parametrize("use_cache", [True, False])
def test_load_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, use_cache: bool
) -> None:
    """Test the _load_dataset function."""
    bids_root: Path = tmp_path / "dataset"
    bids_cache = tmp_path / "cache" if use_cache else None

    mocked_st = MagicMock()
    mocked_st.session_state.get.return_value = {
        "bids_root": str(bids_root),
        "bids_cache": str(bids_cache) if bids_cache else None,
    }

    mocked_load_bids = Mock(return_value=sentinel.layout)
    mocked_parse_layout = Mock(return_value=sentinel.entity_df)

    # Patch the names where _load_dataset looks them up.
    monkeypatch.setattr(mod, "st", mocked_st)
    monkeypatch.setattr(mod, "load_bids", mocked_load_bids)
    monkeypatch.setattr(mod, "parse_layout", mocked_parse_layout)

    mod._load_dataset()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    mocked_load_bids.assert_called_once_with(
        bids_root=Path(bids_root),
        cache_path=bids_cache,
    )
    assert mocked_st.session_state.layout is sentinel.layout

    mocked_parse_layout.assert_called_once_with(sentinel.layout)
    assert mocked_st.session_state.entity_df is sentinel.entity_df

    mocked_st.empty.assert_called_once_with()
    loading_box = mocked_st.empty.return_value
    loading_box.info.assert_called_once_with("Loading BIDS dataset...")
    loading_box.empty.assert_called_once_with()

    mocked_st.toast.assert_called_once_with("BIDS dataset loaded successfully!")


# Test for _handle_query function


@pytest.mark.parametrize(
    ("query", "has_entity_df"),
    [
        (None, False),
        ("", False),
        ("new query", False),
        ("new query", True),
    ],
)
def test_handle_query(
    monkeypatch: pytest.MonkeyPatch,
    query: str | None,
    has_entity_df: bool,
) -> None:
    """Verify query storage, parsing, rendering, and returned results.

    Empty input preserves the previous query. Nonempty input updates it,
    but parsing and rendering occur only when an entity dataframe exists.
    """
    mocked_st: MagicMock = MagicMock()
    mocked_st.text_input.return_value = query

    state: MagicMock = mocked_st.session_state
    state.analysis = {"query": "previous query"}
    state.get.return_value = state.analysis
    state.__contains__.return_value = has_entity_df
    state.entity_df["name"].tolist.return_value = ["subject", "session"]
    state.tbl = sentinel.table

    mocked_parse_query: Mock = Mock(
        return_value=("subject", sentinel.rhs),
    )
    mocked_render: Mock = Mock()

    monkeypatch.setattr(mod, "st", mocked_st)
    monkeypatch.setattr(mod, "parse_query", mocked_parse_query)
    monkeypatch.setattr(mod, "render_inference_choices", mocked_render)

    result: tuple[str | None, object | None] = mod._handle_query()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    mocked_st.text_input.assert_called_once_with(
        "Enter query",
        value="previous query",
        key="query_input",
    )
    assert state.analysis["query"] == (query if query else "previous query")

    if query and has_entity_df:
        mocked_parse_query.assert_called_once_with(
            query,
            ["subject", "session"],
            sentinel.table,
        )
        mocked_render.assert_called_once_with(sentinel.rhs)
        assert result == ("subject", sentinel.rhs)
    else:
        mocked_parse_query.assert_not_called()
        mocked_render.assert_not_called()
        assert result == (None, None)


# Test for _save_analysis_results function


@pytest.mark.parametrize(
    ("valid_outputdir", "verify_space"),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_save_analysis_results(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_outputdir: bool,
    verify_space: bool,
) -> None:
    """Verify results rendering and optional saving.

    A valid output directory enables saving maps, the table, and parameters.
    Space verification additionally generates and records atlas output.
    Saving must preserve inference terms in the original session state.
    """
    lhs: str = "subject"
    statistical_maps: list[str] = ["effect.nii.gz", "statistic.nii.gz"]
    paths: dict[str, str] = {"outputdir": str(tmp_path)}
    analysis: dict[str, object] = {
        "query": "test query",
        "inference_terms": ["term_a"],
        "cluster_size": 10,
    }

    mocked_st: MagicMock = MagicMock()
    state: MagicMock = mocked_st.session_state
    state.paths = paths
    state.get.return_value = paths
    state.analysis = analysis.copy()
    state.result = sentinel.result
    state.masker = sentinel.masker
    state.space_verification_input = verify_space
    state.tbl.describe.return_value = sentinel.summary_stats

    mocked_runner: Mock = Mock()
    mocked_visualization: Mock = Mock()
    mocked_save_maps: Mock = Mock(return_value=statistical_maps)
    mocked_parameter_output: Mock = Mock(
        return_value=sentinel.output_parameters,
    )
    mocked_save_parameters: Mock = Mock()

    def create_atlas_output(
        statistical_map: Path,  # noqa: ARG001
        *,
        cluster_extent: int,  # noqa: ARG001
        outdir: Path,
    ) -> None:
        """Simulate atlas generation by writing an output file."""
        (outdir / "clusters.csv").write_text(
            "cluster,label\n1,example\n",
            encoding="utf-8",
        )

    mocked_create_output: Mock = Mock(side_effect=create_atlas_output)

    monkeypatch.setattr(mod, "st", mocked_st)
    monkeypatch.setattr(mod, "render_model_runner", mocked_runner)
    monkeypatch.setattr(mod, "render_visualization", mocked_visualization)
    monkeypatch.setattr(mod, "save_all_maps", mocked_save_maps)
    monkeypatch.setattr(mod, "create_output", mocked_create_output)
    monkeypatch.setattr(mod, "parameter_output", mocked_parameter_output)
    monkeypatch.setattr(mod, "save_parameters", mocked_save_parameters)

    mod._save_analysis_results(lhs, valid_outputdir)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    # Results are always rendered.
    mocked_runner.assert_called_once_with(lhs)
    mocked_st.subheader.assert_called_once_with("Results")
    state.tbl.describe.assert_called_once_with()
    mocked_st.dataframe.assert_called_once_with(
        sentinel.summary_stats,
        width="content",
    )
    mocked_visualization.assert_called_once_with(["term_a"])

    # Removing inference terms from the saved copy must not alter state.
    assert state.analysis == analysis

    if not valid_outputdir:
        mocked_save_maps.assert_not_called()
        mocked_create_output.assert_not_called()
        state.tbl.to_csv.assert_not_called()
        mocked_parameter_output.assert_not_called()
        mocked_save_parameters.assert_not_called()
        return

    mocked_save_maps.assert_called_once_with(
        tmp_path,
        sentinel.result,
        sentinel.masker,
        lhs,
    )
    state.tbl.to_csv.assert_called_once_with(
        tmp_path / "tbl.csv",
        index=False,
    )

    expected_atlas_output: list[str] | None = None

    if verify_space:
        atlas_output_path: Path = tmp_path / "atlasreader_output"
        assert atlas_output_path.is_dir()
        mocked_create_output.assert_called_once_with(
            tmp_path / "statistic.nii.gz",
            cluster_extent=10,
            outdir=atlas_output_path,
        )
        expected_atlas_output = ["clusters.csv"]
    else:
        mocked_create_output.assert_not_called()
        assert not (tmp_path / "atlasreader_output").exists()

    mocked_parameter_output.assert_called_once_with(
        paths,
        {"query": "test query", "cluster_size": 10},
        statistical_maps,
        "tbl.csv",
        expected_atlas_output,
    )
    mocked_save_parameters.assert_called_once_with(
        tmp_path / "parameters.json",
        sentinel.output_parameters,
    )


# Test for main function


@pytest.fixture
def app_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, dict[str, Mock]]:
    """Replace Streamlit and app helpers with isolated test doubles."""
    mocked_st: MagicMock = MagicMock()
    mocked_st.columns.return_value = (MagicMock(), MagicMock())

    state: MagicMock = mocked_st.session_state
    state.paths = {}
    state.analysis = {}
    state.entity_df = sentinel.entity_df
    state.__contains__.return_value = True

    def click_enabled_button(
        label: str,  # noqa: ARG001
        *,
        disabled: bool = False,
    ) -> bool:
        """Simulate clicking each enabled button."""
        return not disabled

    mocked_st.button.side_effect = click_enabled_button

    helper_names: tuple[str, ...] = (
        "render_header",
        "load_config",
        "render_bids_input",
        "_load_dataset",
        "render_table_input",
        "render_outputdir_input",
        "render_template_input",
        "render_analysis_param_input",
        "render_entity_table",
        "_handle_query",
        "_save_analysis_results",
        "render_footer",
    )
    helpers: dict[str, Mock] = {name: Mock() for name in helper_names}

    helpers["render_bids_input"].return_value = True
    helpers["render_outputdir_input"].return_value = True
    helpers["render_template_input"].return_value = True
    helpers["render_entity_table"].return_value = sentinel.edited_df
    helpers["_handle_query"].return_value = ("subject", sentinel.rhs)

    monkeypatch.setattr(mod, "st", mocked_st)
    for name, mocked_helper in helpers.items():
        monkeypatch.setattr(mod, name, mocked_helper)

    return mocked_st, helpers


@pytest.mark.parametrize("autoload", [False, True])
@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_main_success(
    app_mocks: tuple[MagicMock, dict[str, Mock]],
    tmp_path: Path,
    autoload: bool,
    use_config: bool,
    debug: bool,
) -> None:
    """Verify configuration, UI rendering, loading,.

    analysis, and debug output.
    """
    mocked_st, helpers = app_mocks
    config_file: Path | None = tmp_path / "config.json" if use_config else None
    config: dict[str, dict[str, str]] = {
        "paths": {"bids_root": str(tmp_path / "dataset")},
        "analysis": {"query": "configured query"},
    }
    helpers["load_config"].return_value = config

    mod.main(
        config_file=config_file,
        autoload=autoload,
        debug=debug,
    )

    mocked_st.set_page_config.assert_called_once_with(
        page_title="NeuroVoxel",
        layout="wide",
    )
    helpers["render_header"].assert_called_once_with()
    mocked_st.session_state.setdefault.assert_has_calls(
        [call("paths", {}), call("analysis", {})]
    )

    if use_config:
        helpers["load_config"].assert_called_once_with(config_file)
        assert mocked_st.session_state.paths == config["paths"]
        assert mocked_st.session_state.analysis == config["analysis"]
    else:
        helpers["load_config"].assert_not_called()

    helpers["render_bids_input"].assert_called_once_with(autoload)
    helpers["_load_dataset"].assert_called_once_with()
    helpers["render_table_input"].assert_called_once_with(autoload)
    helpers["render_outputdir_input"].assert_called_once_with(autoload)
    helpers["render_template_input"].assert_has_calls(
        [
            call("Brain template image", "template", autoload),
            call(
                "Binary brain mask specifying voxels to analyze",
                "mask",
                autoload,
            ),
        ]
    )
    helpers["render_analysis_param_input"].assert_called_once_with()

    helpers["render_entity_table"].assert_called_once_with(sentinel.entity_df)
    assert mocked_st.session_state.entity_df is sentinel.edited_df

    helpers["_handle_query"].assert_called_once_with()

    helpers["_save_analysis_results"].assert_called_once_with(
        "subject",
        valid_outputdir=True,
    )
    helpers["render_footer"].assert_called_once_with()

    expected_buttons: list[object] = []
    if not autoload:
        expected_buttons.append(call("Load BIDS dataset", disabled=False))
    expected_buttons.append(call("Run analysis", disabled=False))
    assert mocked_st.button.call_args_list == expected_buttons

    if debug:
        mocked_st.write.assert_called_once_with(mocked_st.session_state)
        mocked_st.expander.assert_any_call("Session State")
    else:
        mocked_st.write.assert_not_called()


@pytest.mark.parametrize(
    ("lhs", "rhs", "valid_mask"),
    [
        (None, None, True),
        ("subject", None, True),
        ("subject", sentinel.rhs, False),
    ],
)
def test_main_skips_analysis(
    app_mocks: tuple[MagicMock, dict[str, Mock]],
    lhs: str | None,
    rhs: object | None,
    valid_mask: bool,
) -> None:
    """Skip loading invalid BIDS input.

    and analysis with missing prerequisites.
    """
    mocked_st, helpers = app_mocks
    mocked_st.session_state.__contains__.return_value = False

    helpers["render_bids_input"].return_value = False
    helpers["render_template_input"].return_value = valid_mask
    helpers["_handle_query"].return_value = (lhs, rhs)

    mod.main()

    mocked_st.button.assert_any_call(
        "Load BIDS dataset",
        disabled=True,
    )
    mocked_st.button.assert_any_call(
        "Run analysis",
        disabled=not (lhs and valid_mask),
    )
    helpers["_load_dataset"].assert_not_called()
    helpers["render_entity_table"].assert_not_called()
    helpers["_save_analysis_results"].assert_not_called()
    helpers["render_footer"].assert_called_once_with()
