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
