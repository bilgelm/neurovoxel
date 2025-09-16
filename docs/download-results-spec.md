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
    <imvar>[_space-<label>]_contrast-<label>_stat-<label>_<mod>map.nii.gz
    ```

    - `<imvar>`: **imaging variable (dependent/LHS)** name used as the filename prefix, sanitized to `[A-Za-z0-9_]`  
      (default). Example: `tau_pet_suvr`.  
      - Optional override: a pipeline/tool prefix (e.g., `ravenmaps`) if explicitly configured.
    - `contrast-<label>`: short contrast/covariate name (CamelCase or short snake_case), e.g., `plasma_p_tau181`, `age`.  
    - `stat-<label>`: one of `t`, `z`, `p`, `F`, `effect`, `variance`, etc. (per BEP041).  
    - `space-<label>` (optional): e.g., `MNI152NLin6Asym`.  
    - `<mod>map`: **modality-aware suffix** (extensible):
        - PET → `petmap`  
        - fMRI → `boldmap`  
        - structural MRI → `anatmap`  
        - **diffusion MRI → `dwimap`**  
        - CT → `ctmap`  
        - (extensible for future modalities)

  - **Examples (BLSA group-level, PET):**
    ```
    tau_pet_suvr_contrast-plasma_p_tau181_stat-t_space-MNI152NLin6Asym_petmap.nii.gz
    tau_pet_suvr_contrast-plasma_p_tau181_stat-p_space-MNI152NLin6Asym_petmap.nii.gz
    ```

  - **Mask** (brain/model mask actually used), with space if relevant:
    ```
    tau_pet_suvr_space-MNI152NLin6Asym_mask.nii.gz
    ```

  - **Notes**
    - Default prefix is the **dependent/LHS imaging variable**; use a pipeline prefix (e.g., `ravenmaps`) only if explicitly requested.
    - Keep `contrast-` labels short and consistent.
    - Follows BEP041 entity order: `[prefix][_space-<label>]contrast-<label>_stat-<label>_<mod>map.nii.gz`.
    - **Sanitization** (for `<imvar>` and `contrast-<label>`): replace non `[A-Za-z0-9_]` with `_`.

- **Tabular summaries (CSV)**:
  - `summary_voxelwise.csv` — run metadata (permutations, thresholds, correction).
  - `clusters.csv` — detailed cluster table (peak MNI x/y/z, max stat, cluster size, corrected p).
  - `results_summary.csv` — **one row per contrast** (clear for multi-contrast runs):
    - `query` *(verbatim user query string — repeated per row)*
    - `imaging_variable` *(LHS/dependent used as filename prefix)*
    - `contrast_name`
    - `stat_peak_t`, `stat_peak_coord_mni_x`, `stat_peak_coord_mni_y`, `stat_peak_coord_mni_z`
    - `stat_peak_p_corrected`, `n_signif_voxels`, `n_clusters`
    - `n_permutations`, `correction_method`, `alpha`
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
    - `dependent` *(LHS/imaging variable name — matches `<imvar>` prefix)*
    - `contrasts: [...]` *(list of tested contrasts, e.g., `["plasma_p_tau181", "age"]`)*
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
  dependent: "tau_pet_suvr"     # becomes filename prefix <imvar>
  contrasts: ["plasma_p_tau181", "age"]
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

  ---

### C. Sidecar metadata

To keep filenames short and clean, while preserving full details, embed the query and run metadata in **`manifest.json`** (and `config.yaml`).

**`manifest.json`** — compact, machine-readable index (per-contrast with paired t/p maps):
```json
{
  "name": "neurovoxel_blsa_tau_pet_suvr_20250912T143210Z",
  "dataset": "BLSA",
  "query": "What is the association of brain tau pathology with plasma p-tau181, adjusting for age?",
  "imaging_variable": "tau_pet_suvr",
  "space": "MNI152NLin6Asym",
  "modality_suffix": "petmap",
  "n_permutations": 1000,
  "correction_method": "TFCE",
  "alpha": 0.05,
  "random_seed": 1729,

  "contrasts": [
    {
      "name": "plasma_p_tau181",
      "statmaps": {
        "t": "tau_pet_suvr_contrast-plasma_p_tau181_stat-t_space-MNI152NLin6Asym_petmap.nii.gz",
        "p": "tau_pet_suvr_contrast-plasma_p_tau181_stat-p_space-MNI152NLin6Asym_petmap.nii.gz"
      },
      "peak": {
        "t": 4.82,
        "coord_mni": [-46, -66, 38],
        "p_corrected": 0.012
      }
    },
    {
      "name": "age",
      "statmaps": {
        "t": "tau_pet_suvr_contrast-age_stat-t_space-MNI152NLin6Asym_petmap.nii.gz",
        "p": "tau_pet_suvr_contrast-age_stat-p_space-MNI152NLin6Asym_petmap.nii.gz"
      },
      "peak": {
        "t": 3.10,
        "coord_mni": [30, -70, 22],
        "p_corrected": 0.041
      }
    }
  ],

  "paths": {
    "mask": "tau_pet_suvr_space-MNI152NLin6Asym_mask.nii.gz",
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

---

## Acceptance criteria

- [ ] Download button disabled until a run completes; then enables with options (Results zip, Config only, Report only).
- [ ] Results zip contains all artifacts listed in this spec.
- [ ] **Filename prefix** is the **dependent/LHS imaging variable** (sanitized). Optional pipeline override is supported.
- [ ] **Map filenames use BEP041 entities**: `contrast-<label>`, `stat-<label>`, optional `space-<label>`, with `<mod>map` suffix.
- [ ] `<mod>map` supports at least: `petmap`, `boldmap`, `anatmap`, `dwimap`, `ctmap`; extensible for future modalities.
- [ ] `results_summary.csv` has **one row per contrast** and includes `imaging_variable` and `contrast_name`.
- [ ] `manifest.json` uses a **`contrasts` array**; each contrast includes paired `statmaps.t` and `statmaps.p` paths and (optionally) peak stats.
- [ ] `config.yaml` is sufficient to exactly rerun the analysis (includes query, parameters, versions, seed).
- [ ] Filenames remain short at the outer (zip/folder) level.
- [ ] No subject-level BLSA data is exported.

---

## Manual test plan

1. Run an example query (*“What is the association of brain tau pathology with plasma p-tau181, adjusting for age?”*).
2. Download results; unzip.
3. Confirm presence of all required files.
4. **Check map filenames** follow the pattern:
   - `<imvar>[_space-<label>]_contrast-<label>_stat-<label>_<mod>map.nii.gz`
   - Prefix equals the dependent/LHS (unless a pipeline override was set).
5. Open the t-map in a viewer; dimensions match template/mask; affine is correct.
6. `results_summary.csv` has **one row per contrast** and includes the `query`, `imaging_variable`, and `contrast_name`.
7. `manifest.json` contains `contrasts[]`; for each contrast, `statmaps.t` and `statmaps.p` exist and correspond to the same `name`.
8. `config.yaml` contains the verbatim query, input paths, parameters, and environment versions.
9. Re-run with the same seed → confirm results reproduce (identical or within tolerance).


