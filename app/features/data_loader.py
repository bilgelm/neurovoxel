from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd
import streamlit as st
from bids.layout.models import BIDSImageFile  # pyright: ignore[reportMissingTypeStubs]
from neurovoxel.io import load_bids
from app.features.query_builder import query_builder


# ---------- Helpers: entity table (unchanged logic + test rows) ----------


def _build_entity_df_from_layout(layout) -> pd.DataFrame:
    """Recreate the image-types table from a BIDSLayout (original app.py behavior)."""
    img_list: List[BIDSImageFile] = layout.get(extension="nii.gz") + layout.get(
        extension="nii"
    )

    img_type_counts: Dict[Tuple[Tuple[str, object], ...], int] = {}
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
        by=["datatype", "suffix", "desc", "param", "trc"], na_position="last"
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
            "Entries in the 'name' column must be unique. Please fix duplicates before continuing."
        )

    return edited_df


# ---------- Helpers: path autodetection ----------


def _detect_bids_config(bids_root: Path) -> Optional[Path]:
    # Try common names anywhere under root (shallow first, then deep)
    candidates = list(bids_root.glob("*entity*config.json")) or list(
        bids_root.rglob("*entity*config.json")
    )
    return candidates[0] if candidates else None


def _detect_template_and_mask(
    bids_root: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
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


def _detect_tabular_from_sibling(bids_root: Path) -> Optional[Path]:
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


def _prefill_blsa_paths() -> Dict[str, str]:
    """Opinionated defaults based on your screenshots/paths."""
    # Adjust these if your BLSA layout moves.
    bids_root = "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging"
    bids_config = "/scratch/hackathon-2025/team-1/blsa_open_entity_config.json"
    tabular = "/scratch/hackathon-2025/team-1/blsa_open_tabular/blsa_open.csv"
    template = "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging/derivatives/template/tpl-blsa_T1w.nii.gz"
    mask = "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging/derivatives/template/tpl-blsa_desc-brain_mask.nii.gz"
    return dict(
        bids_root=bids_root,
        bids_config=bids_config,
        tabular=tabular,
        template=template,
        mask=mask,
    )


def _autodetect_all(
    bids_root_str: str, prefill: bool = False
) -> Dict[str, str]:
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
    # --- Config.yaml upload and validation ---
    import yaml

    st.markdown("### Import config.yaml for reproducibility")
    uploaded_config = st.file_uploader(
        "Upload config.yaml", type=["yaml", "yml"], key="config_yaml_upload"
    )
    config_data = None
    config_error = None
    if uploaded_config:
        try:
            config_data = yaml.safe_load(uploaded_config)
            # Minimal schema validation
            required_top = [
                "bids_root",
                "bids_config",
                "tabular",
                "template",
                "mask",
                "analysis",
                "environment",
                "integrity",
                "notes",
            ]
            missing = [k for k in required_top if k not in config_data]
            if missing:
                config_error = (
                    f"Missing required top-level keys: {', '.join(missing)}"
                )
            # Example: check analysis subkeys
            elif not all(
                k in config_data["analysis"]
                for k in [
                    "dependent",
                    "independent",
                    "covariates",
                    "smoothing_fwhm",
                    "voxel_size",
                    "permutations",
                    "random_seed",
                    "multiple_comparisons",
                ]
            ):
                config_error = "Missing required analysis keys."
            # Example: check environment subkeys
            elif not all(
                k in config_data["environment"]
                for k in ["app_version", "git_commit", "libraries"]
            ):
                config_error = "Missing required environment keys."
            # Example: check integrity subkeys
            elif not all(
                k in config_data["integrity"]
                for k in [
                    "bids_root_hash",
                    "tabular_hash",
                    "template_hash",
                    "mask_hash",
                ]
            ):
                config_error = "Missing required integrity keys."
            # Example: check notes subkeys
            elif "no_raw_blsa_exported" not in config_data["notes"]:
                config_error = (
                    "Missing required notes key: no_raw_blsa_exported."
                )
        except Exception as e:
            config_error = f"YAML parsing error: {e}"
    if uploaded_config:
        if config_error:
            st.error(f"Config file error: {config_error}")
        else:
            st.success("Config file loaded and validated!")
            st.json(config_data)
    """
    Render analysis input form and return user input values.

    Returns (unchanged signature):
        dataset_path, run_clicked, use_bsla_str, bids_config_path

    Side effects:
        - st.session_state.layout: BIDSLayout (when loaded)
        - st.session_state.entity_df: DataFrame of image types (edited)
        - st.session_state.paths: dict with keys bids_root, bids_config, template, mask, tabular
    """
    st.subheader("Dataset")

    testing_mode = st.checkbox(
        "Testing mode (preload example image types)",
        value=False,
        key="testing_mode",
    )
    use_blsa = st.checkbox("Use BLSA data", key="use_blsa")

    # -------- Root path + autodetect --------
    if use_blsa:
        detected = _autodetect_all("", prefill=True)
        st.info(
            "BLSA data selected. All file paths will be loaded automatically."
        )
        bids_config_path = detected["bids_config"]
        template_path = detected["template"]
        mask_path = detected["mask"]
        tabular_path = detected["tabular"]
        # Store all paths for downstream use
        st.session_state.paths = {
            "bids_root": detected["bids_root"],
            "bids_config": bids_config_path,
            "template": template_path,
            "mask": mask_path,
            "tabular": tabular_path,
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
            st.session_state._autodetected = detected  # cache between reruns

        detected = st.session_state.get(
            "_autodetected",
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

        # Store all paths for downstream use
        st.session_state.paths = {
            "bids_root": detected.get(
                "bids_root", st.session_state.get("bids_root_input", "")
            ),
            "bids_config": bids_config_path,
            "template": template_path,
            "mask": mask_path,
            "tabular": tabular_path,
        }

    # -------- Load dataset / testing behavior --------
    # Button disabled in testing mode (mirrors your prior behavior)
    run_clicked = st.button(
        "Load dataset", key="load_dataset_btn", disabled=testing_mode
    )

    if testing_mode:
        entity_df = _build_testing_entity_df()
        st.session_state.entity_df = _render_entity_table(entity_df)
        st.info("Testing mode: using preloaded image types (no dataset read).")
        st.session_state.pop("layout", None)
        image_names = entity_df["name"].drop_duplicates().tolist()
        query_builder(image_names)

        # Return signature compatibility
        use_blsa_str = "bsla" if use_blsa else None
        dataset_path = st.session_state.paths["bids_root"]
        return dataset_path, False, use_blsa_str, bids_config_path

    # Non-testing: load BIDS and build table
    dataset_path = st.session_state.paths["bids_root"]
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

    # -------- Return (unchanged signature) --------
    use_blsa_str = "bsla" if use_blsa else None
    return dataset_path, run_clicked, use_blsa_str, bids_config_path
