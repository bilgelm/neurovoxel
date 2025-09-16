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

- **Tabular summaries (CSV)**:
  - `summary_voxelwise.csv` — run metadata (permutations, thresholds, correction)
  - `clusters.csv` — detailed cluster table (x,y,z peak MNI coords, max stat, cluster size, corrected p)
  - `variables.csv` — dependent, independent, covariates as actually used (resolved names)
  - `results_summary.csv` — **compact one-row summary** for the run:
    - `independent`, `dependent`, `covariates`
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

---

### C. Sidecar metadata
To keep filenames short and clean, while preserving full details:
- **`query.txt`** — verbatim user query text.  
- **`manifest.json`** — compact machine-readable metadata for quick indexing, e.g.:

```json
{
  "name": "neurovoxel_blsa_p-tau181_20250912T143210Z",
  "dataset": "BLSA",
  "independent": "plasma_p_tau181",
  "dependent": "tau_pet_suvr",
  "covariates": ["age","sex"],
  "n_permutations": 1000,
  "correction_method": "TFCE",
  "alpha": 0.05,
  "random_seed": 1729,
  "paths": {
    "stat_map": "stat_map.nii.gz",
    "p_map": "p_map.nii.gz",
    "mask": "mask_used.nii.gz",
    "clusters": "clusters.csv",
    "summary_voxelwise": "summary_voxelwise.csv",
    "results_summary": "results_summary.csv",
    "variables": "variables.csv",
    "config": "config.yaml",
    "report": "report.pdf",
    "query": "query.txt"
  },
  "run_timestamp_iso8601": "2025-09-12T14:32:10Z",
  "app_version": "0.1.0",
  "git_commit": "abc123def456"
}
