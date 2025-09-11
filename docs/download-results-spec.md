# Spec: Downloadable Results + Reproducibility Bundle (Issue #5, BLSA)

## Objective
After a user runs an analysis on BLSA data, provide a **Download** button that exports:
1) statistical results, 2) a **full configuration file** capturing *all* inputs/parameters (for exact reruns), and 3) a human-readable **report**.

No raw subject-level data is exported; only derived files.

---

## Artifacts to export

### A. Results (machine-readable)
- **Statistical map(s)** in analysis space (NIfTI `.nii.gz`):
  - `stat_map.nii.gz` (t- or z-statistics)
  - `p_map.nii.gz` (voxelwise p-values; optional if included elsewhere)
  - `mask_used.nii.gz` (the exact binary mask actually used)
- **Tabular summaries** (CSV):
  - `summary_voxelwise.csv` — run metadata (permutations, thresholds, correction)
  - `clusters.csv` — cluster table (x,y,z peak MNI coords, max stat, size, corrected p)
  - `variables.csv` — dependent, independent, covariates as actually used (resolved names)
- **Provenance**:
  - `VERSION.txt` — app version, git commit SHA, run timestamp, environment summary

### B. Configuration (for exact rerun)
- **`config.yaml`** capturing *everything* needed to rerun:
  - **Data inputs**
    - `bids_root`: absolute path used at run time (BLSA), e.g. `/niaShared/BLSA/BIDS_v1`
    - `tabular_data_path`: e.g. `/niaShared/BLSA/phenotype/participants.tsv`
    - `template_path`, `mask_path`
  - **Variables / model**
    - `dependent`, `independent`, `covariates: [...]`
    - `formula` (e.g., `Y ~ X + age + sex`)
    - `contrast` (if applicable)
  - **Analysis parameters**
    - `smoothing_fwhm_mm`, `voxel_size_mm`
    - `n_permutations`
    - `multiple_comparisons` (method & alpha)
    - `random_seed` (must be captured for reproducibility)
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
bids_root: "/niaShared/BLSA/BIDS_v1"
tabular_data_path: "/niaShared/BLSA/phenotype/participants.tsv"
template_path: "/niaShared/templates/MNI152_2mm_brain.nii.gz"
mask_path: "/niaShared/masks/GM_mask_MNI152_2mm.nii.gz"

variables:
  dependent: "tau_pet_suvr"
  independent: "plasma_p_tau181"
  covariates: ["age", "sex"]
  formula: "tau_pet_suvr ~ plasma_p_tau181 + age + sex"
  contrast: "plasma_p_tau181"

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
