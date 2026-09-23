# Python tau=1 distribution pipeline

This is the Python implementation of the first, deliberately small analysis milestone:

- one-frame lag (`tau_frames = 1`);
- no state switching;
- no bleaching or diffusion-population fit;
- DANAE-like same-frame Inter background;
- one intermediate result per cell;
- one equal-cell-weight result per `(condition, target, genotype)`.

It reproduces the distribution-building section of the postdoc's `TARDIS_v2.m` without requiring SciPy, Pandas, or MATLAB. NumPy is required; Matplotlib is needed only when plots are enabled.

## Run the example

From the repository root:

```bash
PYTHONPATH=path_irrelevant_python python -m tardis_tau1.cli \
  path_irrelevant_python/example_manifest.csv \
  path_irrelevant_python/example_output
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "path_irrelevant_python"
python -m tardis_tau1.cli path_irrelevant_python/example_manifest.csv path_irrelevant_python/example_output
```

## Manifest

One row represents one cell. Required columns:

- `condition_id`
- `target_id`
- `genotype_id`
- `replicate_id`
- `cell_id`
- `input_file`

Optional columns:

- `position_variable` (default `pos`)
- `coordinate_unit` (`m`, `um`, `µm`, or `nm`; default `m`)
- `frame_time_s` (may be blank in this distribution-only stage)
- `enabled` (default true)

Relative input paths are resolved from the manifest directory. Labels are never inferred from filenames.

## Calculation

For each cell, the program computes:

```text
Total                   = normalized f-to-(f+1) pair histogram
InterConditional        = normalized ordered same-frame pair histogram
InterContribution       = beta * InterConditional
IntraContribution       = Total - InterContribution
```

`beta` is the least-squares background scale fitted over the final 30% of bins. The code preserves the legacy behavior of removing every exact-zero same-frame distance. Negative Intra bins are retained and reported as QC warnings rather than silently clipped.

Every cell result and genotype result preserves:

```text
Total = InterContribution + IntraContribution
```

Genotype results average complete cell contributions with equal cell weight. Coordinates and raw pairs are never pooled across cells.

## Outputs

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

The `.npz` files contain portable numeric arrays; JSON contains metadata and QC; CSV is convenient for inspection and plotting.

## Tests

```bash
PYTHONPATH=path_irrelevant_python python -m unittest discover \
  -s path_irrelevant_python/tests -v
```
