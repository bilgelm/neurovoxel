# Spec: Downloadable Results + Reproducibility Bundle (Issue #5, BLSA)

## Objective
After a user runs an analysis on BLSA data, provide a **Download** button that exports:
1) statistical results, 2) a **full configuration file** capturing *all* inputs/parameters (for exact reruns), and 3) a human-readable **report**.

No raw subject-level data is exported; only derived files.

---

## Artifacts to export

### A. Results (machine-readable)

- **Statistical map(s)** (NIfTI `.nii.gz`) named per **BIDS Statistical Model Derivatives (BEP041)**:

  #### Filename convention

  - **Outer bundle (kept short):**
    ```
    neurovoxel_blsa_<indep>_<YYYYMMDDTHHMMSSZ>.zip
    ```

  - **Inner maps (group-level contrast):**
    ```
    dataset-blsa[_space-<label>]_contrast-<label>_stat-<label>_<mod>map.nii.gz
    ```

    - `contrast-<label>`: short contrast name (CamelCase or snake_case).  
    - `stat-<label>`: one of `t`, `z`, `p`, `F`, `effect`, `variance`, etc. (per BEP041).  
    - `space-<label>` (optional): e.g., `MNI152NLin6Asym`.  
    - `<mod>map`: modality-aware suffix:  
      - PET → `petmap`  
      - fMRI → `boldmap`  
      - structural MRI → `anatmap`  
      - CT → `ctmap`

  - **Examples (BLSA group-level):**
    ```
    dataset-blsa_contrast-plasma_p_tau181_stat-t_space-MNI152NLin6Asym_petmap.nii.gz
    dataset-blsa_contrast-plasma_p_tau181_stat-p_space-MNI152NLin6Asym_petmap.nii.gz
    ```

  - **Mask** (brain/model mask actually used), with space if relevant:
    ```
    dataset-blsa_space-MNI152NLin6Asym_mask.nii.gz
    ```

  - **Notes**
    - Use `dataset-blsa` for group-level derivatives from the BLSA dataset.  
    - Keep `contrast-` labels short and consistent.  
    - Follow BEP041 entity order: `[source][_space-<label>]contrast-<label>_stat-<label>_<mod>map.nii.gz`.

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
  
### C. Sidecar metadata

To keep filenames short and clean, while preserving full details, embed the query and run metadata in **`manifest.json`** (and `config.yaml`).

**`manifest.json`** — compact, machine-readable index:
```json
{
  "name": "neurovoxel_blsa_p-tau181_20250912T143210Z",
  "dataset": "BLSA",
  "query": "What is the association of brain tau pathology with plasma p-tau181, adjusting for age?",
  "contrast_name": "plasma_p_tau181",
  "n_permutations": 1000,
  "correction_method": "TFCE",
  "alpha": 0.05,
  "random_seed": 1729,
  "paths": {
    "t_map": "dataset-blsa_contrast-plasma_p_tau181_stat-t_space-MNI152NLin6Asym_petmap.nii.gz",
    "p_map": "dataset-blsa_contrast-plasma_p_tau181_stat-p_space-MNI152NLin6Asym_petmap.nii.gz",
    "mask":  "dataset-blsa_space-MNI152NLin6Asym_mask.nii.gz",
    "clusters": "clusters.csv",
    "summary_voxelwise": "summary_voxelwise.csv",
    "results_summary": "results_summary.csv",
    "config": "config.yaml",
    "report": "report.pdf",
    "version": "VERSION.txt"
  },
  "run_timestamp_iso8601": "2025-09-12T14:32:10Z",
  "app_version": "0.1.0",
  "git_commit": "abc123def456"
}
