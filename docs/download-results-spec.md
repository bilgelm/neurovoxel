# Spec: Downloadable Results + Reproducibility Bundle (Issue #5, BLSA)

## Objective
After a user runs an analysis on BLSA data, provide a **Download** button that exports:
1) statistical results, 2) a **full configuration file** capturing *all* inputs/parameters (for exact reruns), and 3) a human-readable **report**.

No raw subject-level data is exported; only derived files.

---

## Artifacts to export

### A. Results (machine-readable)

- **Statistical map(s)** (NIfTI `.nii.gz`) named per **BIDS Statistical Model Derivatives (BEP041)** / **BIDS Derivatives**:
  - Pattern (group-level contrast map):
    ```
    <source>[_space-<label>]_contrast-<label>_stat-<label>_<mod>map.nii.gz
    ```
    - `contrast-<label>`: short descriptive name for the contrast.
    - `stat-<label>`: one of `t`, `z`, `p`, `F`, `effect`, `variance`, etc. (per BEP041).
    - `space-<label>`: e.g., `MNI152NLin6Asym`, if applicable.
    - `<mod>map`: modality-derived map suffix (e.g., `boldmap`, `petmap`, `anatomicalmap` as appropriate for the pipeline).
  - Examples for our voxelwise regression contrast on PET/tau (group level):
    ```
    task-blsa_contrast-plasma_p_tau181_stat-t_space-MNI152NLin6Asym_petmap.nii.gz
    task-blsa_contrast-plasma_p_tau181_stat-p_space-MNI152NLin6Asym_petmap.nii.gz
    ```
  - **Mask** (brain/model mask actually used), also with space when relevant:
    ```
    task-blsa_space-MNI152NLin6Asym_mask.nii.gz
    ```
  - Notes
    - If no specific “task” applies, use an appropriate `<source>` (e.g., `task-blsa` or `dataset-blsa`) consistently.
    - Keep `contrast-` labels short (CamelCase recommended in BEP041 examples).

- **Tabular summaries (CSV)**:
  - `summary_voxelwise.csv` — run metadata (permutations, thresholds, correction).
  - `clusters.csv` — detailed cluster table (peak MNI x/y/z, max stat, cluster size, corrected p).
  - `results_summary.csv` — **compact one-row summary** (verbatim query kept intact):
    - `query` *(verbatim user query string)*
    - `contrast_name` *(if applicable)*
    - `n_permutations`, `correction_method`, `alpha`
    - `peak_t`, `peak_coord_mni_x`, `peak_coord_mni_y`, `peak_coord_mni_z`
    - `peak_p_corrected`, `n_signif_voxels`, `n_clusters`
    - `smoothing_fwhm_mm`, `voxel_size_mm`, `random_seed`
    - `run_timestamp_iso8601`

- **Provenance**:
  - `VERSION.txt` — app version, git commit SHA, run timestamp, environment summary

---

### B. Configuration (for exact rerun)

- **`config.yaml`** capturing *everything* needed to rerun:

  - **Top-level**
    - `query`: *verbatim user query* (kept as-is; UI does not require splitting)

  - **Data inputs**
    - `bids_root`: absolute path used at run time (BLSA), e.g. `/niaShared/BLSA/BIDS_v1`
    - `tabular_data_path`: e.g. `/niaShared/BLSA/phenotype/participants.tsv`
    - `template_path`, `mask_path`

  - **Variables / model** *(kept for reproducibility; derived from the query by the app)*
    - `dependent`, `contrast`, and any resolved elements (optional if derivable)
    - `formula` if applicable (e.g., `Y ~ X + age`)

  - **Analysis parameters**
    - `smoothing_fwhm_mm`, `voxel_size_mm`
    - `n_permutations`
    - `multiple_comparisons` (method & alpha)
    - `random_seed`

  - **Environment / provenance**
    - `app_version`, `git_commit`, `run_timestamp_iso8601`
    - library versions (`nilearn`, `numpy`, `scipy`)
    - `compute_backend` (CPU/GPU)

  - **Integrity**
    - `file_hashes:` SHA256 for template/mask/tabular
    - `mask_shape`, `template_shape`, `affine_digest`

  - **Privacy**
    - `exports_subject_level_data: false`

**Example (`config.yaml`)**
```yaml
query: "What is the association of brain tau pathology with plasma p-tau181, adjusting for age?"

bids_root: "/niaShared/BLSA/BIDS_v1"
tabular_data_path: "/niaShared/BLSA/phenotype/participants.tsv"
template_path: "/niaShared/templates/MNI152_2mm_brain.nii.gz"
mask_path: "/niaShared/masks/GM_mask_MNI152_2mm.nii.gz"

variables:
  # Optional bookkeeping for reproducibility; primary source of truth is `query`.
  dependent: "tau_pet_suvr"
  contrast: "plasma_p_tau181"
  formula: "tau_pet_suvr ~ plasma_p_tau181 + age"

analysis:
  smoothing_fwhm_mm: 5.0
  voxel_size_mm: 4.0
  n_permutations: 1000
  multiple_comparisons:
    method: "TFCE"
    alpha: 0.05
  random_seed: 1729

provenance:
  app_version: "0.1.0"
  git_commit: "abc123def456"
  run_timestamp_iso8601: "2025-09-11T14:32:10Z"
  libs:
    nilearn: "0.10.4"
    numpy: "2.0.1"
    scipy: "1.13.1"
  compute_backend: "cpu"

integrity:
  file_hashes:
    template_path: "sha256:…"
    mask_path: "sha256:…"
    tabular_data_path: "sha256:…"
  mask_shape: [91, 109, 91]
  template_shape: [91, 109, 91]
  affine_digest: "e3b0c442…"

privacy:
  exports_subject_level_data: false
