# TARDIS/DANAE τ=1 Python pipeline

Tracking-free localization displacement analysis for the first TARDIS milestone:

- one-frame lag only (`tau_frames = 1`);
- MATLAB v5 and MATLAB v7.3/HDF5 input support;
- SI units throughout the Xinran workflow;
- DANAE-like same-frame Inter background;
- independent analysis of every cell;
- equal-cell replica aggregation;
- no state switching, bleaching fit, or diffusion-population fit yet.

The implementation follows the distribution-building logic in the postdoc's MATLAB code while keeping each cell auditable through raw pair counts, normalized distributions, QC metadata, and diagnostic plots.

## Scientific hierarchy

The active Xinran dataset hierarchy is:

```text
condition
  └── target
      └── replica (one MAT file)
          └── cells (Base records T1, T2, ...)
```

Conditions:

- `WT`: wild type;
- `TDK`: TRAK double knockout;
- `MDK`: Miro double knockout.

Targets: `Miro1`, `Miro2`, `TRAK1`, and `TRAK2`.

`genotype` is not part of this active hierarchy. The older manifest/genotype workflow remains available only for backward compatibility.

## Installation

From the repository root:

```bash
python -m pip install -e path_irrelevant_python
```

Main dependencies are NumPy, Matplotlib, and h5py. SciPy and Pandas are not required.

## Calibration and defaults

| Parameter | Value |
|---|---:|
| Frame interval | `0.02 s/frame` |
| Pixel size | `117e-9 m/pixel` |
| Lag | `τ = 1 frame = 0.02 s` |
| Maximum distance | `3e-6 m` |
| Histogram bins | `300` |
| Bin width | `1e-8 m` (10 nm) |
| Tail-fit region | final 30%, bins 211–300 (2.1–3.0 μm) |

Input `x_data` and `y_data` values are converted from pixels to metres. Input time is converted to integer frame IDs using the configured frame interval.

## MATLAB v7.3 record selection

A Xinran MAT file contains three records for each physical cell:

1. Base;
2. `SpotsMaskFiltered`;
3. `TracksFiltered`.

Only Base records are analyzed. Filtered variants are excluded automatically. Every Base record is decoded once in MATLAB cell-array order. Coordinates from different cells are never concatenated, and pair distances are never calculated across cells.

Required Base fields:

```text
name
time
x_data
y_data
tracksMetaData.frameInterval  # validated when present
```

Malformed or duplicate Base records are reported and excluded from replica aggregation.

## Per-cell calculation

### Total at τ=1

For every occupied frame `f`, all localizations in `f` are paired with all localizations in `f+1`:

```text
candidate pairs = Σ_f n_f × n_(f+1)
T_c = H_Total,c / sum(H_Total,c)
```

Distances from `0` through `3e-6 m` are retained. Raw counts are accumulated across the cell and normalized once.

### Same-frame Inter background

For every frame, ordered all-to-all same-frame pairs are calculated:

```text
candidate pairs = Σ_f n_f²
B_c = H_Background,c / sum(H_Background,c)
```

Both `(i,j)` and `(j,i)` are included. Every exact-zero distance is removed to reproduce the legacy MATLAB behavior. Remaining distances through `3e-6 m` are accumulated and normalized once.

### Tail fit and decomposition

```text
beta_raw = (B_tailᵀ T_tail) / (B_tailᵀ B_tail)
beta     = clip(beta_raw, 0, 1)
Inter_c  = beta_c × B_c
Intra_c  = T_c - Inter_c
T_c      = Inter_c + Intra_c
```

`beta_raw`, constrained `beta`, and the clipping flag are preserved. Negative Intra bins are retained as QC information; they are never clipped, smoothed, or replaced by absolute values.

## Replica aggregation

The canonical replica result is **cell-balanced**. Every valid cell has equal weight:

```text
Total_rep = mean(T_c)
Inter_rep = mean(beta_c × B_c)
Intra_rep = mean(Intra_c)
beta_rep  = mean(beta_c)
```

The Inter calculation is explicitly `mean(beta_c × B_c)`, not `mean(beta_c) × mean(B_c)`.

No pair-weighted replica Total, beta, Inter, or Intra distribution is produced. Per-cell raw pair counts remain available for QC and audit.

## Probability density and positive Intra summaries

The canonical arrays remain per-bin probability masses. Bin-width-independent densities are also saved in inverse micrometres:

```text
width_um[i] = (edge[i+1] - edge[i]) × 1e6
density_um_inv[i] = probability[i] / width_um[i]
```

Therefore `sum(density_um_inv × width_um)` recovers the corresponding probability mass. Signed Intra density retains every negative bin and is never smoothed, clipped, or converted to an absolute value.

For interpretation, each cell and replica also receives a `positive_intra` summary computed from `max(Intra, 0)` without changing canonical signed Intra. It reports the peak position and density, total positive mass, positive mass within 0–300 nm, and positive-mass-weighted mean and median distance. The 300 nm boundary uses fractional bin overlap, and the median is linearly interpolated within its crossing bin. Replica summaries are computed from the replica mean Intra curve, not by averaging nonlinear cell summaries.

## Reproducibility provenance

Cell and replica JSON results record the complete configuration, input SHA-256 and byte size, analysis runtime, schema version, package version, Python version, Git commit, and Git dirty state. One MAT replica is fingerprinted once and that identity is shared by all of its Base cells. Git or installed-package metadata may be `null` when unavailable; missing provenance never prevents scientific analysis. Output serialization time is excluded from the recorded analysis runtime.

## Running one Base cell

The default pilot runs `C2-101625_WT_Miro1_T1`:

```bash
PYTHONPATH=path_irrelevant_python \
python path_irrelevant_python/run_xinran_pilot.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "path_irrelevant_python"
python path_irrelevant_python/run_xinran_pilot.py
```

## Running one complete replica

The default runner analyzes all 10 Base cells in `WT / Miro1 / 20251016_WT_Miro1_V7.mat`:

```bash
PYTHONPATH=path_irrelevant_python \
python path_irrelevant_python/run_xinran_replica.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "path_irrelevant_python"
python path_irrelevant_python/run_xinran_replica.py
```

Options:

```text
--input <replica.mat>
--output <output-directory>
--condition <WT|TDK|MDK>
--target <Miro1|Miro2|TRAK1|TRAK2>
--replica <replica-id>
--expected-cells <count>
--bins <count>              # default: 300 (10 nm/bin over 3 μm)
--no-plot
```

## Project directory

```text
path_irrelevant_python/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── run_xinran_pilot.py       # one named Base cell
├── run_xinran_replica.py     # all Base cells in one MAT replica
├── tardis_tau1/
│   ├── __init__.py
│   ├── config.py             # τ=1 parameters and validation
│   ├── matio.py              # MATLAB v5 and v7.3 readers
│   ├── core.py               # cell analysis and replica aggregation
│   ├── outputs.py            # JSON/NPZ/CSV/PNG serialization
│   ├── provenance.py         # hashes, versions, runtime, and Git state
│   ├── batch.py              # legacy manifest/genotype workflow
│   └── cli.py                # legacy manifest CLI
└── tests/
    ├── test_tau1.py
    ├── test_replica.py
    └── test_enhancements.py
```

## Replica output directory

```text
replica_output/
└── <condition>/
    └── <target>/
        └── <replica>/
            ├── replica_result.json
            ├── replica_result.npz
            ├── replica_distributions.csv
            ├── cell_summary.csv
            ├── replica_summary.csv
            ├── replica_distribution.png
            ├── all_cell_intra.png
            └── cells/
                └── <cell-id>/
                    ├── cell_result.json
                    ├── cell_result.npz
                    ├── distributions.csv
                    └── diagnostic.png
```

### Replica files

- `replica_result.json`: identifiers, parameters, counts, probability/density distributions, positive-Intra summary, QC, and full provenance.
- `replica_result.npz`: numeric arrays including probabilities, densities, beta, positive-Intra scalars, provenance scalars, `cell_ids`, and both cell Intra matrices (`n_cells × n_bins`).
- `replica_distributions.csv`: equal-cell replica probability masses and densities in SI distance bins.
- `cell_summary.csv`: per-cell beta, localization/pair counts, signed-negative QC, and positive-Intra summaries.
- `replica_summary.csv`: one-row replica scientific summary plus schema, input hash, versions, Git state, and runtime.
- `replica_distribution.png`: density curves in full-range and 0–0.5 μm panels.
- `all_cell_intra.png`: every cell's unmodified signed Intra density plus the thick equal-cell replica mean, in full and zoom panels.

The replica CSV does not invent a raw pooled frequency for the equal-cell mean. Real raw frequencies remain in each cell's `distributions.csv` and NPZ file.

### Cell distribution CSV columns

```text
distance_left_m
distance_right_m
distance_mid_m
total_count
inter_background_count
total_probability
inter_background_probability
inter_contribution_probability
intra_contribution_probability
inter_expected_count
intra_expected_count
total_density_um_inv
inter_background_density_um_inv
inter_contribution_density_um_inv
intra_contribution_density_um_inv
intra_conditional_density_um_inv
```

Raw Total and same-frame background counts come from different pair pools and must not be directly subtracted. Expected Inter/Intra counts place the probability contributions on the retained Total-pair scale; they may be fractional, and Intra expected counts may be negative.

## First replica validation

Validated replica: `WT / Miro1 / 20251016_WT_Miro1_V7`.

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
| Replica negative Intra bins | 134 |
| Positive Intra peak | 0.055 μm |
| Positive Intra mass | 0.1745450818 |
| Positive Intra mass, 0–300 nm | 0.1650372922 |
| Positive-weighted mean distance | 0.1463261920 μm |
| Positive-weighted median distance | 0.0913423954 μm |

The enhanced result uses cell schema `tau1-cell-v3` and replica schema `tau1-replica-cell-balanced-v4`. The input SHA-256 begins `765effd0a438`, and all 10 cells share the same full fingerprint. Density integration recovers Total mass 1, Inter mass 0.8302165316, and Intra mass 0.1697834684.

The saved `cell_intra_contributions` array has shape `10 × 300`; its binwise mean exactly equals the saved replica Intra contribution. Compared with the 30 nm baseline, beta changed by only `-0.0003605`; the narrower bins expose more signed bin-to-bin residual variation, increasing negative-bin count from 42 to 134 and negative mass from `0.003586` to `0.004762`.

## QC interpretation

A `pass_with_warnings` result can still satisfy all reconstruction and normalization identities. Typical warnings include negative Intra bins, coincident same-frame detections removed, missing frames, sparse Inter tails, beta clipping, and excluded malformed cells.

Negative Intra bins are diagnostic residuals, not automatically a processing failure.

## Tests

```bash
PYTHONPATH=path_irrelevant_python \
python -m unittest discover -s path_irrelevant_python/tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "path_irrelevant_python"
python -m unittest discover -s path_irrelevant_python/tests -v
```

The current suite contains 21 tests covering pair semantics, MATLAB readers, SI conversion, beta constraints, equal-cell aggregation, density integration, positive-Intra summaries, provenance, two-panel plots, output round trips, and the `testPos2.mat` regression.

## Legacy manifest workflow

The original version-1 manifest CLI remains available for reproducibility:

```bash
PYTHONPATH=path_irrelevant_python \
python -m tardis_tau1.cli \
  path_irrelevant_python/example_manifest.csv \
  path_irrelevant_python/example_output
```

It uses the older condition/target/genotype/replicate/cell hierarchy and should not be confused with the active Xinran replica workflow.

## Current limitations

- Only `τ=1` is supported.
- Only one replica is processed per active runner invocation.
- The full 47-file dataset has not been batch processed.
- State switching is not fitted.
- Bleaching kinetics are not fitted from this single lag.
- Diffusion coefficients and multi-population Brownian models are not fitted.
- Cross-replica and condition-level aggregation are not implemented.
- Negative Intra bins are retained rather than forced to zero.
