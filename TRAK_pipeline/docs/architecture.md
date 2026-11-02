# Architecture and scientific cache contracts

## Public boundary

`tools/launch.py` chooses one immutable common release; platform launchers only locate the existing runtime and forward a command/configuration. The shared CLI exposes `plan`, `run`, `summarize --completed-only`, `refresh --completed-only` and a stop-after-current-cell request. Mac and Windows use the same stage implementations and independently configured executables.

`StageContext` supplies the resolved dataset/sample, parameters for one stage, named upstream `Artifact` objects, owned output directory and runtime paths. A stage returns named files and explicit metadata in `StageResult`. It does not publish another stage's completion marker or modify an upstream file.

The artifact ID is calculated from `(method_id, method_version, canonical scientific parameters, upstream IDs)`. Paths, release timestamps and source-file hashes are provenance, not scientific identity. Source and dependency checksums are independently frozen for publication integrity. Scientific changes must update the responsible method version; decorative changes do not.

## Dependency chain

```text
C2 → detection → fit ───────────────────────────────┐
ROI ───────────→ fit                                │
C1 → broad → support counts → threshold → cleanup ─→ assignment
ROI ──────────────────────────────────────→ mask ROI │
C1 original zeros + full-image mask + ROI ──────────┘
fit + UID labels → distances → decomposition → bootstrap
C1 → arithmetic mean
saved scientific statistics + mean + analysis products → presentation
completed cell scientific results → BR/condition statistics → summary presentation
```

Full-image Nellie does not depend on ROI. Support-count identity contains the temporal window, stride, actual requested participants and counting rule; threshold identity contains the saved counts and threshold. Cleanup uses full-image FG8 components and enclosed BG4 holes. ROI intersection is its own operation and cannot manufacture a red contour along an ROI cut.

The broad stack is stored losslessly in compressed bit-packed chunks. Existing older boolean NPY stacks are read with bounded mmap slices instead of materializing a full movie or rerunning segmentation. Counts and final masks remain separate products. Execution chunk scheduling and worker count do not define a biological method.

## Spatial and temporal contract

A complete mask artifact describes TYX shape, original frame IDs, frame-to-block mapping, actual and expected participating frames, block validity/reasons, pixel-center convention, pixel/time units, crop offset and C1/C2 registration. Default coordinates are integer pixel centers with closed unit squares and zero crop offset; identity registration is an explicit unverified assumption.

Missing original frames are not zero masks. An incomplete support block is unknown for affected original frames. A complete block with an empty mask is a valid empty observation. The consumer validates coordinate and unit consistency before geometry, and fails on contradictions rather than inferring a transform. Original same-frame C1 zero pixels and closed-FOV boundaries retain their unknown precedence.

## Localization and UID contract

The persisted detection source identity and native detection ID define UID. Sorting, refined coordinates and assignment labels cannot define it. The legacy reader checks UID uniqueness, row/detection correspondence, original frame/time units, selected-point QC, and full observed-Hessian X/Y standard-error semantics. Missing meaning is reported explicitly.

Fit results remain one read-only master table. Assignment products contain UID-keyed labels and reasons. Downstream analysis reads the current assignment's declared label field; it must not accidentally reuse a historical label column already present in a legacy master.

Legacy adapters translate saved parameter contracts and completed artifacts without moving or rewriting old results. An unavailable intermediate can be materialized from compatible upstream data when an explicit run requires it. Refresh does not materialize missing scientific intermediates. Uncertain legacy compatibility is reported, never replaced by an automatic refit.

## Distances and bootstrap

Distance counts depend on the coordinates/valid UID set, UID labels, original endpoint frame pairs, τ, complete bin edges and pair/zero-distance rules. The v1 method explicitly accepts only τ=1. Label or endpoint selection changes invalidate affected group counts.

Retained products include per-original-frame-anchor Total/background counts, zero-removal statistics, validity mappings, source/target frame IDs, resampling strata and frame-block membership. Shared weights, replicate IDs, common valid masks and compact scalars are persisted. Full-movie aggregate histograms alone cannot reproduce the temporal bootstrap. No artificial adjacent pair is introduced between resampled blocks.

## Scientific statistics versus rendering

Filtering, normalization, histogram bins, membership and cell/BR weighting belong to scientific statistics. Their outputs are persisted before rendering. Fonts, labels, colors, layout and column order belong to presentation. Normalization is not silently performed inside a refresh operation.

`refresh --completed-only` has a hard zero-scientific-call boundary, including when an input is missing. It reports the missing product and does not fall through to `run` or `summarize`. `summarize --completed-only` explicitly computes hierarchy statistics from completed cell results, then publishes them; it never initiates missing localization, segmentation or single-cell TARDIS.

Group membership is method-specific. Different scientific methods cannot silently share a summary roster. Missing required mask frames make the full-movie area mean unavailable, rather than silently dropping acquisition time; an invalid first block has no first-area estimate. Existing On%, first/time-mean ROI mask area, per-cell counts and pooled-cell/equal-replica localization means retain their existing definitions. Scientific completion and publication completion are separate, so a missing figure or workbook does not discard completed fitting or segmentation.

## Area hierarchy scientific method v2

For the area view, normalize each eligible cell's signed Intra by that cell's
signed area on the full 0–3 um grid, then calculate the arithmetic mean within
BR. At condition level, average those corrected BR area means with one weight
per BR. Do not substitute a pooled-cell mean across BRs, normalize the raw BR
signed mean, or normalize the final unit-area mean again.

The scientific area branch is:

    saved cell signed Intra
      -> cell signed-area normalization
      -> equal-cell BR area mean
      -> equal-BR condition area mean
      -> saved scientific summary -> presentation

V1's BR normalize(mean(raw signed curves)) is superseded for this view. With
positive cell areas it was an area-weighted mean of normalized shapes. V2 gives
each eligible cell the same shape weight within BR.

The explicit hierarchy method version is 2. Its scientific parameters are
area_order="normalize_cells_then_equal_cell_mean" and
condition_area_order="equal_BR_mean_of_corrected_area_means". Summary/input/numeric
identities change independently of source hashes; unchanged single-cell artifact
IDs are reused. Rendering reads the saved corrected mean and cannot silently
repair or recalculate a v1 scientific product.

The signed branch still averages signed curves. Existing peak rules remain
unchanged. The common valid roster, guards, negative bins, unavailable-value
rules and between-BR sample SD/SEM are retained. On%, ROI-mask areas and
localization counts keep their separate metric definitions.

## Storage, interruption and migration

All controlled writes, scratch/cache directories, logs and state stay on the external drive. Writes are atomic with unique temporary names; independent phases have separate state files. Writable memmaps are closed before committed metadata is captured. Existing protected-path, subprocess cancellation, failure continuation and clean-stop behavior are retained.

Migration proceeds through shared I/O and read-only legacy adapters, then independent mask/assignment/cache stages, then analysis/statistics/presentation and finally thin launchers. Source relocation itself does not change a scientific ID. The 211 existing TRAK cells are reused, not recomputed during migration. Old releases and launcher backups remain available for rollback.

Acceptance uses small fixtures plus the ten fixed T5 frames. It checks cache invalidation boundaries, missing-frame/UID/unit failures, original-frame resampling boundaries, and zero scientific calls during both successful and missing-input refresh. It does not add a native Mac/Windows parity qualification gate or a production pilot.
