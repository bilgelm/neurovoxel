# NeuroVoxel

_NeuroVoxel_ is a Python-based application for quantifying associations of
neuroimaging measures with non-imaging variables.

## Requirements

_NeuroVoxel_ requires Python 3.13+ and the following modules:
`nilearn` for neuroimage I/O, visualization, and statistical analysis,
`nanslice` for dual-coded statistical map visualization,
`streamlit` for the web app, and
`click` for the CLI.

## 💾 Installation

You can install _NeuroVoxel_ via pip after cloning the repository:

```bash
git clone https://github.com/bilgelm/neurovoxel.git
pip install -e neurovoxel
```

## 🚀 Usage

From the command line:

```bash
python -m neurovoxel-app

# or

neurovoxel-app
```

### JSON configuration

Copy [config.template.json](config.template.json), replace the example paths
and query, then launch:

```bash
neurovoxel-app --config-file config.template.json
```

Add `--autoload` to load the BIDS dataset automatically and hide valid path
inputs. Running the analysis still requires the **Run analysis** button.
`--autoload` and `--debug` are command-line flags, not JSON settings.
All JSON settings are optional; omit any you want to enter in the app.
Relative paths resolve from the working directory where you launch the app.

The `paths` object supports these inputs:

| Key | Input |
| --- | --- |
| `bids_root` | BIDS derivatives directory |
| `bids_cache` | Optional existing BIDS parquet cache file; leave empty or omit to skip |
| `tabular` | CSV or TSV covariates file containing `subject` and `session` columns |
| `template` | Brain template image used as a visualization background |
| `mask` | Binary brain mask image specifying voxels to analyze |
| `outputdir` | Empty output directory, or a directory the app can create |

The `analysis` object supports these settings. The template uses the current
defaults except for its example query:

| Key | Default | Accepted values |
| --- | --- | --- |
| `query` | No query | Formula such as `gm ~ age + sex`; replace `gm` with a name from the image types table and use covariates from your table |
| `smoothing_fwhm` | `5.0` | Number from 0 to 15, in mm |
| `voxel_size` | `4.0` | Number from 1 to 10, in mm, isotropic |
| `correction` | `"fwe"` | `"fwe"`, `"fdr"` |
| `n_perm` | `10000` | Integer from 0 to 100000; used for FWE |
| `tfce` | `false` | Boolean; used for FWE |
| `handle_zero_voxels` | `"keep"` | `"keep"`, `"exclude"` |
| `handle_multiple_sessions` | `"first"` | `"first"`, `"last"`, `"random"`, `"all"` |
| `voxelwise_standardization` | `"scale"` | `"scale"`, `"center"`, `"none"` |
| `n_jobs` | `-1` | Nonzero integer; `-1` uses all CPUs; configuration only |
| `random_state` | `42` | Integer random seed; configuration only |

Use decimal JSON numbers for `smoothing_fwhm` and `voxel_size`, as shown in
the template, to match the Streamlit numeric controls. Mixed-effects queries
automatically select `correction: "fdr"` and
`handle_multiple_sessions: "all"`.

Some settings cannot currently be prefilled from JSON: image names in the
editable image types table, inference-term checkboxes, the **Run AtlasReader?**
checkbox, and its cluster-size inputs. Set these in the app. Inference terms
default to all non-intercept model terms.

The internal schema (`src/neurovoxel/utils/template.json`) is a validation
schema, not a runnable configuration example. It lists `random_seed` and
`bids_config`, which the app does not use. Use `analysis.random_state` for
the random seed. The app also reads `paths.bids_cache` and `analysis.n_jobs`,
although these are not explicitly listed in that schema; validation allows
additional properties.

### Multiple comparison correction

Choose **FWE** (default) or **FDR** under **Advanced → Multiple comparison
correction**, or set `analysis.correction` to `"fwe"` or `"fdr"` in a JSON
configuration file. Existing configurations default to FWE.

FWE uses permutation max-t correction, with optional TFCE. FDR uses
[Benjamini-Hochberg][bh-fdr]
correction of two-sided parametric Student t p-values across all retained
voxels,
separately for each tested term. As with `permuted_ols`, tested terms are fitted
independently, adjusting for the non-tested terms and an intercept. Permutation
count and TFCE are ignored for FDR.

Plots use the selected correction. FDR-adjusted p-value maps are saved as
`stat-logpfdr_map.nii.gz` (with the imaging outcome and contrast prefixes),
containing `-log10(adjusted p)`. The selected method is recorded in
`parameters.json`.

### Longitudinal mixed-effects models

Add a random-effects term to the query:

```text
gm ~ age + sex + (1 + age | subject)
```

This fits a random intercept and age slope for each subject, with an estimated
covariance between them. Use `(1 | subject)` for a random intercept or
`(0 + age | subject)` for a random slope without a random intercept. The grouping
column can have another name, such as `id`, provided it exists in the table.
One grouping column and one random-effects term are currently supported;
multiple, nested, crossed, and `||` structures are rejected.

Mixed models automatically retain all sessions in the app. Missing fixed or
random covariates and missing group labels are excluded together. All fixed
effects are fitted jointly using Statsmodels `MixedLM` with REML; the inference
checkboxes select which fixed effects to report. The app selects FDR and
disables permutation controls. When calling `run_query` directly, use
`correction="fdr"`; permutation FWE for mixed models is not supported.

Each voxel explicitly tries L-BFGS, BFGS, then CG, using a fresh model for each
attempt and 100 iterations per attempt. Exceptions, nonconvergence, and
nonfinite fixed-effect inference trigger the next optimizer. The first
converged fit with finite beta/z/p is accepted; warnings are retained for
review. Exhausted retries receive NaN beta/z and p=1 before FDR.

When an output directory is selected, mixed fits also save:

- `lme_fit_summary.json`: failure reasons, warning categories and examples,
  optimizer outcomes, recovered voxel counts, model specification and settings.
- `lme_fit_diagnostics.jsonl`: one record per voxel with all optimizer attempts,
  exact warnings and repetition counts, exception messages and tracebacks,
  elapsed times, available iteration counts, random-effect covariance estimates
  and residual variances. Records include the zero-based voxel index, IJK
  coordinates and world coordinates in millimeters in the final fitted mask.

These files are written after voxel fitting and before plotting or raising an
all-voxel inference error, so failed runs retain their diagnostics. Successful
runs reference the files in `parameters.json`. Without an output directory,
`run_query` returns diagnostics in `result["mixed_diagnostics"]`; an all-voxel
failure raises `MixedModelFitError` with a `diagnostics` attribute. Warning
occurrence counts include every attempt; terminal failure counts exclude
voxels recovered by retries.

Mixed queries can also use age-based terms without precomputing columns:

```text
gm ~ baseline_age + time + (time | subject)
```

With a numeric `age` column and a `subject` column, `baseline_age` is each
subject's minimum age **after image matching and all row exclusions**. Both
`time` and `time_from_baseline` equal `age - baseline_age`, in the same units as
`age`. Baseline uses minimum age, independent of row or session-label order,
and always groups by `subject`, even if random effects use another group.
Baseline and elapsed time are recalculated if formula transforms exclude
additional visits. Visits missing age or subject are excluded when these terms
are used. The retained table saved as `tbl.csv` contains the derived columns.

These three names are reserved computed terms in mixed queries: existing
columns with those names are replaced by the calculated values when any of
these terms is requested. Query previews generate term names only; actual
fitting recalculates the values from the matched, retained visits.

Mixed-model inference uses two-sided asymptotic Wald z-tests, without
small-sample Satterthwaite or Kenward-Roger adjustments. Their uncorrected
p-values use the
same Benjamini-Hochberg correction across voxels per reported term as OLS.
Plots and exports include fixed-effect beta, z, FDR and convergence maps.
Convergence maps contain 1 for successful inference and 0 for failures; failed
voxels have NaN beta/z and p=1 before FDR. A run stops if every voxel fails.
Numerical warnings are summarized in a warning. Voxelwise mixed fitting can be
slow; `analysis.n_jobs` controls parallel fitting. A progress bar appears
immediately while images are prepared, then reports completed voxel fits,
including unsuccessful fits. Estimates of random effects
are fitted separately at each voxel and are not exported.

#### Model fit indicators

Mixed fits return one value per retained voxel for each of these indicators,
independent of which fixed effects are selected for reporting:

- `r_square`: Nakagawa **marginal R²**, the fraction of model variance explained
  by fixed effects.
- `r_square_conditional`: Nakagawa **conditional R²**, the fraction explained
  by fixed and random effects together.
- `f_stat`: an **asymptotic Wald F statistic** jointly testing that all
  non-intercept fixed effects are zero. With `q` tested coefficients,
  `F = beta.T @ inv(cov(beta)) @ beta / q` for that coefficient subset.
- `f_pvalues`: uncorrected omnibus p-values from the equivalent chi-squared
  test with `q` degrees of freedom, or F with `q` and infinite denominator
  degrees of freedom. These do not use OLS residual degrees of freedom or
  small-sample Satterthwaite/Kenward-Roger approximations. The omnibus test
  does not test the random effects and does not replace per-term FDR results.

R² uses the sample variance of fixed predictions `X beta`, the mean
visit-specific random variance `mean(diag(Z cov_re Z.T))`, and the fitted
residual variance. This includes random-slope variance and intercept/slope
covariance, following [Johnson's random-slope extension][mixed-r2]. The R²
denominator is the sum of those three variances; the marginal numerator is
fixed variance, and the conditional numerator is fixed plus random variance.
These variance-partition R² values differ from OLS residual-sum-of-squares R².

Failed fits receive NaN fit indicators. Intercept-only fixed models have no
omnibus slope test and receive NaN `f_stat` and `f_pvalues`. A nonfinite or
non-positive-definite contrast covariance also leaves the omnibus test NaN
without changing per-term inference or the convergence map.

With an output directory, each indicator is saved once per model as a 3D map,
such as `gm_stat-rsquare_map.nii.gz`, `gm_stat-rsquareconditional_map.nii.gz`,
`gm_stat-fstat_map.nii.gz`, and `gm_stat-fpvalues_map.nii.gz`. Maps preserve NaN
at failed voxels. Maps and histograms are included in `lme_results.pdf`;
unavailable indicators are skipped. OLS continues to use `ols_results.pdf`.
The mixed fit summary records the metric definitions and test degrees of
freedom.

[mixed-r2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4368045/

[bh-fdr]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_NeuroVoxel_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@carderne]'s [Postmodern Python template].

[@carderne]: https://github.com/carderne
[Postmodern Python template]: https://github.com/carderne/postmodern-python
[file an issue]: https://github.com/bilgelm/neurovoxel/issues

<!-- github-only -->

[license]: LICENSE
[contributor guide]: CONTRIBUTING.md
