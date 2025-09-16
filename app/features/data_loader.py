"""Data loader and config validation for NeuroVoxel app."""
from __future__ import annotations
from copy import deepcopy

from pathlib import Path
import pandas as pd
import streamlit as st
import yaml
from typing import Any, TYPE_CHECKING

from app.features.query_builder import query_builder
from neurovoxel.io import load_bids

if TYPE_CHECKING:
    from bids.layout.models import BIDSImageFile  # pyright: ignore[reportMissingTypeStubs]

# ---------- Helpers: entity table (unchanged logic + test rows) ----------


def _build_entity_df_from_layout(layout: "BIDSLayout") -> pd.DataFrame: 
    """Recreate the image-types table from a BIDSLayout (original app.py behavior).

    Args:
        layout (BIDSLayout): The BIDS layout object.

    Returns:
        pd.DataFrame: DataFrame of image types.
    """
    img_list: list[BIDSImageFile] = (
        layout.get(extension="nii.gz") + layout.get(extension="nii")
    )
    img_type_counts: dict[tuple[tuple[str, object], ...], int] = {}
    entity_df = pd.DataFrame()

    for img in img_list:
        entities = deepcopy(img.entities)
        for k in ("subject", "session"):
            entities.pop(k, None)
        key = tuple(entities.items())
        if key in img_type_counts:
            img_type_counts[key] += 1
        else:
            entity_df = pd.concat(
                [entity_df, pd.DataFrame([entities])], ignore_index=True
            )
            img_type_counts[key] = 1

    entity_df = entity_df.drop(
        ["SpatialReference", "extension", "tracer"], axis=1, errors="ignore"
    )
    entity_df = entity_df.sort_values(
        by=["datatype", "suffix", "desc", "param", "trc"],
        na_position="last"
    ).reset_index(drop=True)

    def concat_name(row: pd.Series) -> str:
        parts = [
            str(row[col])
            for col in ["desc", "param", "trc", "meas", "suffix"]
            if col in row and pd.notna(row[col])
        ]
        return "_".join(parts) if parts else "Enter name here"

    entity_df["name"] = entity_df.apply(concat_name, axis=1)
    return entity_df


def _build_testing_entity_df() -> pd.DataFrame:
    """Rows matching your screenshot with associated entities."""
    rows = [
        ("wml_mask", "anat", "wml", "blsa", "mask", None, None, None),
        ("csf_ravensmap", "anat", "csf", "blsa", "ravensmap", None, None, None),
        ("gm_ravensmap", "anat", "gm", "blsa", "ravensmap", None, None, None),
        ("vn_ravensmap", "anat", "vn", "blsa", "ravensmap", None, None, None),
        ("wm_ravensmap", "anat", "wm", "blsa", "ravensmap", None, None, None),
        ("ad_dwimap", "dwi", None, "blsa", "dwimap", "ad", None, None),
        ("fa_dwimap", "dwi", None, "blsa", "dwimap", "fa", None, None),
        ("md_dwimap", "dwi", None, "blsa", "dwimap", "md", None, None),
        ("rd_dwimap", "dwi", None, "blsa", "dwimap", "rd", None, None),
        ("pib_dvr_mimap", "pet", None, "blsa", "mimap", None, "dvr", "pib"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "datatype",
            "desc",
            "space",
            "suffix",
            "param",
            "meas",
            "trc",
        ],
    )


def _render_entity_table(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Editable table: only 'name' is editable; validates uniqueness."""
    st.write("Types of images in dataset:")

    disabled_cols = [c for c in entity_df.columns if c != "name"]
    cols = list(entity_df.columns)
    if "name" in cols:
        cols.remove("name")
    cols = ["name", *cols]

    edited_df = st.data_editor(
        entity_df,
        disabled=disabled_cols,
        key="entity_table_editor",
        use_container_width=True,
        column_order=cols,
        hide_index=True,
    )

    if "name" in edited_df and edited_df["name"].duplicated().any():
        st.error(
            "Entries in the 'name' column must be unique. "
            "Please fix duplicates before continuing."
        )

    return edited_df


# ---------- Helpers: path autodetection ----------


def _detect_bids_config(bids_root: Path) -> Path | None:
    # Try common names anywhere under root (shallow first, then deep)
    candidates = list(bids_root.glob("*entity*config.json")) or list(
        bids_root.rglob("*entity*config.json")
    )
    return candidates[0] if candidates else None


def _detect_template_and_mask(bids_root: Path) -> tuple[Path | None, Path | None]:
    tmpl_dir = bids_root / "derivatives" / "template"
    template = None
    mask = None
    if tmpl_dir.is_dir():
        # Template (T1w in name is common in your examples)
        t_candidates = list(tmpl_dir.glob("*tpl-*T1w*.nii*"))
        template = t_candidates[0] if t_candidates else None
        # Brain mask
        m_candidates = list(tmpl_dir.glob("*desc-brain_mask*.nii*"))
        mask = m_candidates[0] if m_candidates else None
    return template, mask


def _detect_tabular_from_sibling(bids_root: Path) -> Path | None:
    # Looks for a sibling folder named *_tabular/ with a single main CSV
    parent = bids_root.parent
    if parent.is_dir():
        tabular_dirs = [
            p
            for p in parent.iterdir()
            if p.is_dir() and p.name.endswith("_tabular")
        ]
        for d in tabular_dirs:
            csvs = list(d.glob("*.csv"))
            if csvs:
                return csvs[0]
    return None


def _prefill_blsa_paths() -> dict[str, str]:
    """Opinionated defaults based on your screenshots/paths."""
    # Adjust these if your BLSA layout moves.
    bids_root = "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging"
    bids_config = "/scratch/hackathon-2025/team-1/blsa_open_entity_config.json"
    tabular = "/scratch/hackathon-2025/team-1/blsa_open_tabular/blsa_open.csv"
    template = (
        "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging/derivatives/template/"
        "tpl-blsa_T1w.nii.gz"
    )
    mask = (
        "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging/derivatives/template/"
        "tpl-blsa_desc-brain_mask.nii.gz"
    )
    return {
        "bids_root": bids_root,
        "bids_config": bids_config,
        "tabular": tabular,
        "template": template,
        "mask": mask,
    }


def _autodetect_all(bids_root_str: str, *, prefill: bool = False) -> dict[str, str]:
    """Return a dict of string paths, detected or empty if not found."""
    if prefill:
        return _prefill_blsa_paths()

    out = {
        "bids_root": bids_root_str,
        "bids_config": "",
        "tabular": "",
        "template": "",
        "mask": "",
    }
    if not bids_root_str:
        return out

    root = Path(bids_root_str)
    if not root.is_dir():
        return out

    cfg = _detect_bids_config(root)
    tmpl, msk = _detect_template_and_mask(root)
    tab = _detect_tabular_from_sibling(root)

    out["bids_config"] = str(cfg) if cfg else ""
    out["template"] = str(tmpl) if tmpl else ""
    out["mask"] = str(msk) if msk else ""
    out["tabular"] = str(tab) if tab else ""
    return out


# ---------- Main UI entry ----------


def data_loader() -> tuple[str, bool, str | None, str | None]:
    """Main UI entry for loading data and config in NeuroVoxel app.

    Returns:
        tuple[str, bool, str | None, str | None]:
            (dataset_path, run_clicked, use_blsa_str, bids_config_path)
    """
    st.markdown("### Import config.yaml for reproducibility")
    config_path = Path.cwd() / "config.yaml"
    config_data = None
    config_error = None
    imported_new_config = False

    # File uploader for new config.yaml
    uploaded_file = st.file_uploader(
        "Import a new config.yaml",
        type=["yaml", "yml"],
        key="config_upload"
    )
    if uploaded_file is not None:
        try:
            config_data = yaml.safe_load(uploaded_file)
            config_error = validate_config_yaml(config_data)
            imported_new_config = True
        except (yaml.YAMLError, TypeError) as e:
            config_error = f"YAML parsing error: {e}"
    else:
        # Use default config.yaml from workspace root
        try:
            with config_path.open() as f:
                config_data = yaml.safe_load(f)
            config_error = validate_config_yaml(config_data)
        except (FileNotFoundError, yaml.YAMLError, TypeError) as e:
            config_error = f"YAML parsing error: {e}"

    if config_error:
        st.error(f"Config file error: {config_error}")
        return "", False, None, None

    # If config is valid, set config_analysis in session state for query_builder prefill
    if config_data and "analysis" in config_data:
        st.session_state["config_analysis"] = config_data["analysis"]
        # Track config source for UI captions
        if imported_new_config:
            st.session_state["config_source"] = "uploaded"
        else:
            st.session_state["config_source"] = "default"
    else:
        st.session_state["config_analysis"] = {}
        st.session_state["config_source"] = "default"

    # Only show success message if a new config was imported
    if imported_new_config:
        st.success("Config file loaded and validated!")
        st.json(config_data)

    # --- Render rest of UI (dataset input, entity table, etc.) ---
    # Determine testing mode from config
    testing_mode_default = False
    if config_data and "environment" in config_data:
        testing_mode_default = bool(config_data["environment"].get("testing", False))

    # UI Inputs
    testing_mode, use_blsa = render_dataset_inputs(testing_mode=testing_mode_default)
    handle_autodetect(use_blsa=use_blsa, testing_mode=testing_mode)
    run_clicked = st.button(
        "Load dataset", key="load_dataset_btn", disabled=testing_mode
    )

    if testing_mode:
        return handle_testing_mode()

    # Non-testing: load BIDS and build table
    dataset_path = st.session_state.paths["bids_root"]
    bids_config_path = st.session_state.paths["bids_config"]
    handle_bids_loading(
        run_clicked=run_clicked,
        dataset_path=dataset_path,
        bids_config_path=bids_config_path,
    )
    use_blsa_str = "bsla" if use_blsa else None
    return dataset_path, run_clicked, use_blsa_str, bids_config_path

def validate_config_yaml(config_data: dict[str, Any]) -> str | None:
    """Validate the config.yaml schema for NeuroVoxel.

    Returns error string if invalid, else None.
    """
    error_msgs: list[str] = []
    required_top = [
        "bids_root",
        "bids_config",
        "tabular",
        "template",
        "mask",
        "analysis",
        "environment",
        "notes",
    ]
    missing = [k for k in required_top if k not in config_data]
    if missing:
        error_msgs.append(f"Missing required top-level keys: {', '.join(missing)}")

    # Check analysis subkeys
    analysis_keys = [
        "query",
        "smoothing_fwhm",
        "voxel_size",
        "permutations",
        "random_seed",
        "multiple_comparisons",
    ]
    analysis: dict[str, Any] = config_data.get("analysis", {})
    missing_analysis = [k for k in analysis_keys if k not in analysis]
    if missing_analysis:
        error_msgs.append(
            f"Missing required analysis keys: {', '.join(missing_analysis)}"
        )

    # Check environment subkeys
    environment: dict[str, Any] = config_data.get("environment", {})
    env_keys = ["testing"]
    missing_env = [k for k in env_keys if k not in environment]
    if missing_env:
        error_msgs.append(
            f"Missing required environment keys: {', '.join(missing_env)}"
        )

    # Check notes subkeys
    notes: dict[str, Any] = config_data.get("notes", {})
    notes_keys = ["no_raw_blsa_exported"]
    missing_notes = [k for k in notes_keys if k not in notes]
    if missing_notes:
        error_msgs.append(
            f"Missing required notes keys: {', '.join(missing_notes)}"
        )

    if error_msgs:
        return "\n".join(error_msgs)
    return None

"""
Render analysis input form and return user input values.

Returns (unchanged signature):
    dataset_path, run_clicked, use_bsla_str, bids_config_path

Side effects:
    - st.session_state.layout: BIDSLayout (when loaded)
    - st.session_state.entity_df: DataFrame of image types (edited)
    - st.session_state.paths: dict with keys bids_root,
      bids_config, template, mask, tabular
"""

def render_dataset_inputs(*, testing_mode: bool) -> tuple[bool, bool]:
    """Render dataset input UI.

    Args:
        testing_mode (bool): Whether testing mode is enabled.

    Returns:
        tuple[bool, bool]: (testing_mode_val, use_blsa)
    """
    st.subheader("Dataset")
    testing_mode_val = st.checkbox(
        "Testing mode (preload example image types)",
        value=testing_mode,
        key="testing_mode",
        disabled=testing_mode,
    ) if testing_mode else st.checkbox(
        "Testing mode (preload example image types)",
        value=False,
        key="testing_mode",
    )
    use_blsa = st.checkbox("Use BLSA data", key="use_blsa")
    return testing_mode_val, use_blsa

def handle_autodetect(*, use_blsa: bool, testing_mode: bool) -> None:
    """Handle autodetection of BIDS paths.

    Args:
        use_blsa (bool): Whether to use BLSA data.
        testing_mode (bool): Whether testing mode is enabled.
    """
    if use_blsa:
        detected = _autodetect_all("", prefill=True)
        st.info(
            "BLSA data selected. All file paths will be loaded automatically."
        )
        st.session_state.paths = {
            "bids_root": detected["bids_root"],
            "bids_config": detected["bids_config"],
            "template": detected["template"],
            "mask": detected["mask"],
            "tabular": detected["tabular"],
        }
    else:
        bids_root_input = st.text_input(
            "BIDS root directory",
            key="bids_root_input",
            placeholder="/path/to/.../blsa_open_neuroimaging",
        )
        if st.button(
            "Auto-detect files from root",
            disabled=not bids_root_input or testing_mode,
            key="autodetect_btn",
        ):
            detected = _autodetect_all(bids_root_input, prefill=False)
            st.session_state.autodetected = detected
        detected = st.session_state.get(
            "autodetected",
            {
                "bids_root": bids_root_input,
                "bids_config": "",
                "tabular": "",
                "template": "",
                "mask": "",
            },
        )
        col1, col2 = st.columns(2)
        with col1:
            bids_config_path = st.text_input(
                "Optional: BIDS config file",
                value=detected.get("bids_config", ""),
            )
            template_path = st.text_input(
                "Brain template image (NIfTI)",
                value=detected.get("template", ""),
            )
        with col2:
            mask_path = st.text_input(
                "Brain mask image (NIfTI)", value=detected.get("mask", "")
            )
            tabular_path = st.text_input(
                "Tabular CSV (e.g., clinical data)",
                value=detected.get("tabular", ""),
            )
        st.session_state.paths = {
            "bids_root": detected.get(
                "bids_root", st.session_state.get("bids_root_input", "")
            ),
            "bids_config": bids_config_path,
            "template": template_path,
            "mask": mask_path,
            "tabular": tabular_path,
        }

def handle_testing_mode() -> tuple[str, bool, str | None, str | None]:
    """Handle UI and logic for testing mode.

    Returns:
        tuple[str, bool, str | None, str | None]: Dataset path, run_clicked,
        use_blsa_str, bids_config_path.
    """
    entity_df = _build_testing_entity_df()
    st.session_state.entity_df = _render_entity_table(entity_df)
    st.info("Testing mode: using preloaded image types (no dataset read).")
    st.session_state.pop("layout", None)
    image_names = entity_df["name"].drop_duplicates().tolist()
    query_builder(image_names)
    dataset_path = st.session_state.paths["bids_root"]
    use_blsa_str = "bsla" if st.session_state.get("use_blsa", False) else None
    bids_config_path = st.session_state.paths.get("bids_config", "")
    return (
        dataset_path,
        False,
        use_blsa_str,
        bids_config_path
    )

def handle_bids_loading(
    *, run_clicked: bool, dataset_path: str, bids_config_path: str
) -> None:
    """Handle loading of BIDS dataset and building entity table.

    Args:
        run_clicked (bool): Whether load button was clicked.
        dataset_path (str): Path to BIDS dataset.
        bids_config_path (str): Path to BIDS config file.
    """
    if run_clicked:
        if not dataset_path:
            st.error("Please enter a BIDS root directory.")
        else:
            root = Path(dataset_path)
            if not root.is_dir():
                st.error(f"Directory does not exist: {dataset_path}")
            else:
                st.info("Loading BIDS layout…")
                layout = load_bids(
                    bids_root=root,
                    config_fname=Path(bids_config_path)
                    if bids_config_path
                    else None,
                )
                st.session_state.layout = layout
                st.success("Dataset loaded successfully!")
                entity_df = _build_entity_df_from_layout(layout)
                st.session_state.entity_df = _render_entity_table(entity_df)
                image_names = entity_df["name"].drop_duplicates().tolist()
                query_builder(image_names)
