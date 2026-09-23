# Changelog

All notable changes to the Python tau=1 distribution pipeline are documented here.

## [0.1.0] - 2026-09-23

### Added

- Added a Python implementation of the first TARDIS/DANAE analysis milestone under `path_irrelevant_python/`.
- Added tau=1-frame analysis for one localization cell at a time.
- Added Total JPP construction from all frame `f` to frame `f+1` localization pairs.
- Added a DANAE-like Inter background from ordered same-frame localization pairs.
- Added legacy-compatible removal of all exact-zero same-frame distances.
- Added tail-based Inter scaling using the final 30% of histogram bins.
- Added the three primary contribution outputs:
  - `Total`
  - `InterContribution`
  - `IntraContribution`
- Added normalized `InterConditional` and `IntraConditional` distributions.
- Added an explicit binwise reconstruction guarantee:

  ```text
  Total = InterContribution + IntraContribution
  ```

- Added QC reporting for:
  - negative Intra bins;
  - negative Intra mass;
  - background fractions outside `[0, 1]`;
  - missing frames;
  - coincident same-frame detections;
  - sparse background tails;
  - normalization and reconstruction errors.
- Added a dependency-light MATLAB v5 MAT-file reader, allowing `testPos2.mat` to be read without SciPy.
- Added chunked NumPy pair-distance histogram accumulation to avoid storing all pair distances.
- Added CSV manifest-driven batch processing with one row per cell.
- Added hierarchical identifiers for condition, target, genotype, biological replicate, and cell.
- Added equal-cell-weight genotype aggregation without pooling coordinates across cells.
- Added replicate-stratified cell bootstrap support for genotype-level confidence intervals.
- Added portable output formats:
  - compressed NumPy `.npz` arrays;
  - JSON metadata and QC;
  - CSV distributions and cell summaries;
  - PNG diagnostic plots.
- Added a command-line interface through `python -m tardis_tau1.cli`.
- Added an example manifest and usage documentation.
- Added unit, regression, aggregation, and batch round-trip tests.

### Validated

- Ran the Python pipeline successfully on `testPos2.mat`.
- Reproduced the postdoc MATLAB distribution logic with the following regression values:

  | Metric | Value |
  |---|---:|
  | Localizations | 189,590 |
  | Frames | 2,500 |
  | Total candidate pairs | 16,370,405 |
  | Total retained pairs | 520,113 |
  | Inter candidate pairs | 16,477,456 |
  | Inter zero distances removed | 189,632 |
  | Inter retained pairs | 425,388 |
  | Background fraction β | 0.8177098334831252 |
  | Intra fraction | 0.1822901665168748 |
  | Negative Intra bins | 40 |
  | Minimum Intra contribution | -0.0006306613549036041 |

- Confirmed that Python and MATLAB results agree to floating-point precision.
- Confirmed that all four Python tests pass.
- Rendered and inspected both cell-level and genotype-level diagnostic plots.

### Output structure

```text
output/
  cells/<condition>/<target>/<genotype>/<replicate>/<cell>/
    cell_result.npz
    cell_result.json
    distributions.csv
    diagnostic.png
  genotypes/<condition>/<target>/<genotype>/
    genotype_result.npz
    genotype_result.json
    distributions.csv
    cell_summary.csv
    diagnostic.png
  failures.csv
```

### Current limitations

- Only `tau_frames = 1` is supported.
- State switching and changing-state kinetics are not fitted.
- Diffusion coefficients and multi-population Brownian models are not fitted yet.
- Bleaching kinetics are not estimated from a single lag.
- Motion-blur correction is not needed or implemented at this distribution-only stage.
- MATLAB v7.3/HDF5 MAT files are not supported by the dependency-free MAT reader.
- Negative Intra bins are retained as diagnostic information rather than clipped.
- A genotype containing only one cell produces a point estimate but no interpretable biological confidence interval.
- The example `testPos2.mat` output uses `UNASSIGNED` metadata because genotype and target labels are not contained in the MAT file.
