# Changelog

All notable changes to the Python τ=1 distribution pipeline are documented here.

## [Unreleased]

### 2026-09-28

#### Added

- Added biological-replica group aggregation for every `condition / target`
  combination. Group means are the equal-weight mean of replica curves; cell
  counts do not determine a replica's weight.
- Added pointwise sample standard deviation (SD) and standard error of the mean
  (SEM) across biological replicas for Total, Inter, and Intra contribution
  densities.
- Added `group_result.npz`, `group_result.json`, group summary CSVs, and group
  figures for all three distributions.
- Added Intra area-normalized (conditional probability-density) curves at the
  cell, replica, group, and comparison levels. Each signed curve is divided by
  its net Intra mass (`1 - beta`), integrates to one, and retains negative bins;
  positive Intra mass remains a separate summary quantity.
- Added cell-level peak-normalized Intra overlays and shape summaries.
- Added distribution mode, mean, and median summaries to area-normalized
  comparison data and figures.
- Added target-level condition comparisons in the fixed order `WT`, `TDK`,
  `MDK`, plus condition-level comparisons of `Miro1`, `Miro2`, `TRAK1`, and
  `TRAK2`.
- Added generic input-directory discovery and the
  `run_xinran_all_groups.py` batch entry point. It supports the active data
  layout `input_root / target / condition / *.mat` without hard-coded file
  names.
- Added `run_xinran_group.py` for one target/condition and
  `view_cells_6_7_timepoints.m` for MATLAB timepoint scatter inspection of
  records 6 and 7.
- Added group-analysis tests and expanded replica-output tests for the new
  normalized figures.
- Added peak-normalized, 0–0.5 μm Intra comparison figures and matching CSV
  data for all four target-level and all three condition-level comparisons.
- Added bin-width-independent probability densities in μm⁻¹ while preserving
  every probability and count output.
- Added positive-Intra peak, mass, 0–300 nm mass, weighted mean, and
  interpolated median summaries for cells and replicas.
- Added full-range and 0–0.5 μm density panels to cell, replica, and all-cell
  Intra plots.
- Added SHA-256 input fingerprints, configuration, schema/package/Python
  versions, analysis runtime, Git commit, and dirty-state provenance.
- Added `replica_summary.csv` and provenance/summary scalars to NPZ output.
- Added tests for density conversion and integrals, positive-Intra summaries,
  provenance, two-panel plots, and enhanced output round trips.
- Added group schema `tau1-group-replica-balanced-v2` with ordered replica
  hashes, shared config/calibration, source environments, and strict rejection
  of incompatible or duplicate biological-replica inputs.
- Added separate `replica_metric_summary.csv` and
  `group_mean_curve_summary.csv` outputs so mean-of-replica metrics cannot be
  confused with metrics calculated from the displayed group mean curve.
- Added SHA-256/config-driven replica resume manifests, atomic completion,
  strict result/artifact validation, legacy v4 adoption, cache counts in the
  batch summary, and `--force-recompute`.

#### Changed

- Replaced 95% t-confidence-interval shading in group plots with the requested
  pointwise `mean ± 1 SD` band.
- The 0–0.5 µm zoom panel of cell, replica, and group plots now shows only the
  Intra curve; the full-range panel still shows Total, Inter, and Intra.
- Increased primary curve line widths for clearer exported figures.
- Replaced every generated full-range/zoom dual-panel PNG with two independent
  files using the suffixes `_full_range.png` and `_zoom_0p5um.png`.
- Standardized generated output under `TARDIS/results/` as
  `target / condition / {replicas, group, comparison}` plus a
  `condition_comparison/` directory. Earlier trial folders are retained only
  as local, regenerable artifacts.
- Updated schemas to `tau1-cell-v3` and `tau1-replica-cell-balanced-v4`.
- Extended the README with probability-density definitions, positive-Intra
  metrics, provenance fields, and the enhanced output tree.
- Corrected area-normalization terminology to signed net Intra mass and added
  explicit rejection of incompatible distance grids or invalid normalization
  denominators in comparison outputs.
- Corrected the group-runner description from t-based uncertainty to the actual
  equal-replica sample SD/SEM and mean ± 1 SD plot bands.
- Unified comparison medians with the canonical within-bin interpolation used
  by positive-Intra summaries, including nonuniform histogram bins.
- Group runners now hash each MAT file once, reuse validated replica results,
  and always regenerate group/comparison derivatives from current code.

#### Removed

- Stopped generating legacy combined dual-panel PNG names such as
  `diagnostic.png`, `replica_distribution.png`, `all_cell_intra.png`,
  `intra_contribution.png`, and `intra_area_normalized.png`. They are replaced
  by explicit full-range and zoom file names.

#### Validation

- Re-ran the complete active dataset: 12 target/condition groups, 47 biological
  replicas, and 539 included cells.
- Generated all cell-, replica-, group-, target-comparison, and
  condition-comparison PNG/CSV/NPZ outputs. The batch manifest is written to
  `TARDIS/results/batch_summary.csv`.
- Earlier in the same date's work, revalidated
  `WT / Miro1 / 20251016_WT_Miro1_V7` after adding the derived outputs:

| Metric | Result |
|---|---:|
| Included cells | 10 |
| Equal-cell replica beta | 0.830216531615378 |
| Positive Intra peak | 0.055 μm |
| Positive Intra mass | 0.1745450818 |
| Positive Intra mass, 0–300 nm | 0.1650372922 |
| Positive-weighted mean distance | 0.1463261920 μm |
| Positive-weighted median distance | 0.0913423954 μm |

- Density integrals recover Total mass 1, Inter mass 0.8302165316, and Intra mass 0.1697834684.
- All 10 cells share the same MAT SHA-256, beginning `765effd0a438`; the validation run recorded commit `fc2b3306c99aa4ec25f2a01b478a22de36891ac7` with a dirty working tree.
- Confirmed that the enhanced scientific arrays are identical to the preceding 10 nm output.
- Confirmed that all 35 tests pass, including direct comparison, cache,
  strict group compatibility, and input-discovery coverage.
- Rendered and inspected the full-range/zoom versions of cell `diagnostic.png`, `replica_distribution.png`, and `all_cell_intra.png`.

### 2026-09-27

#### Added

- Added MATLAB v7.3/HDF5 input support through `h5py`.
- Added one-pass discovery of all unfiltered Base records in one MAT replica.
- Added SI calibration for Xinran data: `0.02 s/frame`, `117e-9 m/pixel`, and distance output in metres.
- Added one-cell and complete-replica Xinran runners.
- Added constrained beta fitting while preserving `beta_raw` and clipping status:

  ```text
  beta = clip(beta_raw, 0, 1)
  ```

- Added raw Total and same-frame background histogram counts to cell NPZ and CSV output.
- Added Total-scale expected Inter and Intra counts to cell CSV output.
- Added canonical equal-cell replica aggregation:

  ```text
  Total_rep = mean(T_c)
  Inter_rep = mean(beta_c × B_c)
  Intra_rep = mean(Intra_c)
  beta_rep  = mean(beta_c)
  ```

- Added replica JSON, NPZ, CSV, cell-summary, and diagnostic outputs.
- Added `all_cell_intra.png`, showing every valid cell's unmodified Intra contribution and the equal-cell replica mean.
- Added `cell_ids` and `cell_intra_contributions` to replica NPZ output.
- Added tests for Base discovery, malformed/duplicate records, beta constraints, exact equal-cell aggregation, pair-count conservation, and replica output round trips.

#### Changed

- Corrected the active hierarchy to:

  ```text
  condition → target → replica (one MAT file) → cells
  ```

- Defined active conditions as WT, TDK, and MDK.
- Removed `genotype` from the active Xinran replica schema; the older genotype manifest remains only as a legacy workflow.
- Standardized the canonical replica result on equal-cell weighting.
- Changed the default histogram from 100 bins (30 nm/bin) to 300 bins (10 nm/bin) while retaining the 3 μm maximum distance and 2.1–3.0 μm beta-fit tail.
- Removed the experimental pair-weighted replica decomposition after first-replica review.
- Retained per-cell raw pair counts for QC without using them to weight the replica distribution.
- Replaced the former dual-aggregation files with `replica_distributions.csv`, `replica_distribution.png`, and `all_cell_intra.png`.
- Preserved signed negative Intra bins instead of clipping them.
- Rewrote the README around the active Xinran workflow and documented the project/output directory structures.

#### Removed

The active replica output no longer produces:

- `pair_weighted_distributions.csv`;
- `cell_balanced_distributions.csv`;
- `mode_comparison.json`;
- `aggregation_comparison.png`;
- pair-weighted replica beta or pair-concentration interpretation.

#### Validation

##### One-cell pilot

Validated `C2-101625_WT_Miro1_T1`:

| Metric | Result |
|---|---:|
| Localizations | 200,576 |
| Frames | 2,300 |
| Time range | 4.00–49.98 s |
| Total retained pairs | 872,352 |
| Background retained pairs | 700,172 |
| Beta | 0.8012321630849819 |
| Negative Intra bins | 134 |

Expected-count reconstruction was verified to approximately `7.3e-12` pairs.

##### First complete replica

Validated `WT / Miro1 / 20251016_WT_Miro1_V7`:

| Metric | Result |
|---|---:|
| Discovered Base cells | 10 |
| Included cells | 10 |
| Excluded cells | 0 |
| Total candidate pairs | 234,361,315 |
| Total retained pairs | 10,103,360 |
| Background candidate pairs | 234,848,099 |
| Background retained pairs | 8,584,250 |
| Equal-cell replica beta | 0.830216531615378 |
| Reconstruction error | 6.94e-18 |
| Negative Intra bins | 134 |

- With the 10 nm histogram, the saved `cell_intra_contributions` array has shape `10 × 300`, and its binwise mean equals the saved replica Intra contribution exactly.
- Compared with the 30 nm baseline, beta changed by `-0.0003605`; negative Intra mass increased from `0.003586` to `0.004762` as the narrower bins exposed more binwise residual variation.

## [0.1.0] - 2026-09-23

### Added

- Added the initial Python implementation under `path_irrelevant_python/`.
- Added τ=1 Total construction from all frame `f` to frame `f+1` localization pairs.
- Added a DANAE-like Inter background from ordered same-frame localization pairs.
- Added legacy-compatible removal of every exact-zero same-frame distance.
- Added tail-based Inter scaling over the final 30% of histogram bins.
- Added Total, Inter contribution, and residual Intra contribution outputs.
- Added exact binwise reconstruction:

  ```text
  Total = InterContribution + IntraContribution
  ```

- Added QC for negative Intra bins/mass, missing frames, coincident detections, sparse tails, normalization, and reconstruction.
- Added a dependency-light MATLAB v5 numeric MAT reader.
- Added chunked NumPy pair-distance histogram accumulation.
- Added the original CSV manifest batch workflow and equal-cell genotype aggregation.
- Added NPZ, JSON, CSV, and PNG outputs.
- Added the original command-line interface and regression tests.

### Validation

Validated `testPos2.mat` against the postdoc MATLAB logic:

| Metric | Result |
|---|---:|
| Localizations | 189,590 |
| Frames | 2,500 |
| Total candidate pairs | 16,370,405 |
| Total retained pairs | 520,113 |
| Inter candidate pairs | 16,477,456 |
| Inter zero distances removed | 189,632 |
| Inter retained pairs | 425,388 |
| Beta | 0.8177098334831255 |
| Negative Intra bins | 40 |
| Minimum Intra contribution | -0.0006306613549036076 |

Python and MATLAB agreed to floating-point precision.
