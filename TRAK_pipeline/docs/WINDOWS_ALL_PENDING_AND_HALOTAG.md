# Windows pending target batch and separate HaloTag-only control

## Start

With the entire My Book drive attached to Windows, double-click `START_ALL_PENDING_WINDOWS.cmd` in the drive root. No Setup, dependency installation, Mac/Windows numerical comparison or pilot is scheduled. Existing Windows Python, Fiji, Java and the four dependency environments are reused.

Execution order:
1. Continue the original TRAK1/TRAK2 BR4/BR5 queue from its existing state and stage checkpoints. The declared queue is 156 cells. Three cells were already complete at preparation; 153 remain. The existing 211 BR1–BR3 cells are summary-only. Five-replica summaries use all 186 TRAK1 and 181 TRAK2 cells, with unavailable results explicit.
2. Process the independent 20-cell HaloTag-only queue: WT, BR1 and BR2, ten cells each. The completed BR1 T1 pilot is adopted only after checking its source identities, method family and committed output presence. It is reused; the remaining 19 cells are queued.

At preparation, 172 cells remain across both jobs. The launcher does not merge scientific records, curves or statistics between jobs. Completion counts are recalculated at runtime.

## Pause and resume

Double-click `STOP_AFTER_CURRENT_CELL_ALL_WINDOWS.cmd` to finish the current cell and then stop the entire combined batch. If the stop arrives between jobs, the next job will not start. Double-click `RESUME_ALL_PENDING_WINDOWS.cmd` (or the Start file again) to continue. A new explicit Start/Resume clears the previous stop request.

Ctrl+C interrupts the active batch and its owned child processes. Committed checkpoints remain available. Do not unplug the drive until the active console has exited. Duplicate launches are blocked with ownership-aware locks; the legacy target-batch locks are also respected. The old Mac and target-only Windows launchers remain available.

Cell-specific failures are recorded and processing continues. A disconnected drive, write failure, shared environment failure or other infrastructure stop prevents the next job from starting. Summary publication failures remain explicit and can be retried without recomputing cell science.

## Speed settings (2026-11-06)

From 2026-11-06 the batch is pinned to `releases/20261106T_pending_and_Halo_controls_v5_roicheck`. `WT_TRAK1_BR5_T5`, which failed in assignment on 2026-11-05, is retried automatically and should now complete. This release only changes how work is scheduled and stopped; the results are the same as before. If a stage fails, its running fit workers and Nellie chunks stop at once, and the drive is released promptly. Completed cells and checkpoints are reused. A cell that was paused mid-stage continues from its last committed chunk. The expected time is about 20 minutes per cell, down from about 61. That figure is estimated, not yet measured on Windows.

At the start of a cell the console should show these lines:

- `"fit_scheduler", "workers": 10`, meaning `fit_workers: "auto"`.
- `"Nellie broad scheduler", "chunk_workers": 3`.
- `TrackMate threads 12` and `maximum 625 frames`. These appear when a cell starts detection.

To change a count without editing files, set an environment variable before starting: `TRAK_FIT_WORKERS`, `TRAK_DETECTION_THREADS` or `TRAK_NELLIE_WORKERS`. If TrackMate runs out of Java heap, the batch stops with its checkpoints kept; set `TRAK_DETECTION_THREADS=2` and resume. Details and rollback steps are in the README section "Speed settings — 2026-11-06".

## HaloTag-only output locations

All cell figures remain beside each cell's C1, C2 and ROI in `Analysis_Figures_modular_<ID>/`, with an English index. Cell scientific products and checkpoints remain in the same cell's `analysis_results/modular_cache/`.

Summary figures and the reader workbook are placed in `Analysis_Figures_Cell_Metrics/` at each completed cell, BR folder, `NegativeControl/WT/`, and the outer negative-control data root. Intra figures are in `Analysis_Figures_Intra/`, with same-axis overlays in its `On_NotOn_Overlay/` subfolder. Intra source data, metric CSVs, membership, progress, parameters and audit records are stored in the corresponding `analysis_results/modular_summary_<ID>/`. The separate top-level root is `Xinran_TRAK1_TRAK2_negativeWT`.

HaloTag-only is a HaloTag-only construct with fluorescent labeling and no TRAK1/TRAK2 fusion. It is not a dye-only or untransfected control. Its records and figures are never added to either target's summaries. No new control-versus-target significance test or randomization analysis is added by this batch.

## Metrics and averaging

For every cell, save the QC-passed total localization count, On, Not-on, unknown, QC-excluded and ROI-candidate counts. On percent is `100 × On / (On + Not-on)`; unknown and QC-excluded points remain outside that denominator.

The first-mask area is the original cell ROI intersected with the first 50-frame support mask (frames 0–49). Mean mask area is the time-weighted mean ROI area across the 50 support blocks covering frames 0–2499. Units are square micrometers. This is mask foreground area, not an individual-mitochondrion object size.

BR summaries list the cells and use equal-cell means. WT/dataset summaries show pooled-cell and equal-BR means separately. Main count summaries are per-cell means, not sums over cells. Numeric tables retain full-precision mean, sample SD, SEM and participating cell/BR counts. The reader workbook `Cell_Metrics.xlsx` uses the existing concise template with cell values, means, SD and participation counts; SEM is retained in the numerical summary tables. Technical source IDs and paths are saved separately from the reader workbook.

Intra summaries keep the corrected cell-first normalization: each valid cell's signed Intra is divided by its own signed area before equal-cell averaging within a BR; BR means are then averaged equally. The final mean is not normalized again. Signed and peak views, shared On/Not-on eligibility, negative bins, source data, full-range and zoom views, and same-axis On/Not-on overlays are retained. Progress summaries state the full expected roster, including pending cells. No missing curve is drawn as zero.

## Saved-results-only actions

`UPDATE_ALL_SUMMARIES_WINDOWS.cmd` explicitly calculates summaries from saved cell products and exports figures and Excel. It does not run localization, segmentation, classification or single-cell TARDIS.

`REFRESH_ALL_FIGURES_WINDOWS.cmd` only publishes already saved summary products. If scientific summary data are missing, it reports them and does not compute them implicitly. `PLAN_ALL_PENDING_WINDOWS.cmd` prints the queues and reusable completion counts without starting cells.

## Preserved methods and files

Original C1/C2/ROI files and historical results remain unchanged. The default raw-C1 Nellie broad → support ≥10/50 → post-min32 → enclosed holes <64 workflow and original 1-SE ellipse assignment remain unchanged. Localization, full observed-Hessian nominal SE, QC, τ=1 TARDIS and shared 1000-replicate time-block bootstrap retain the established parameters. No pixel-distance assignment or contrast-stretched C1 test becomes a batch default.

Adaptive C1 display uses pooled nonzero ROI percentiles from ten fixed representative frames, then a single range within each cell. It changes display only. C2 display stays fixed. Nominal calibration and unverified identity registration labels remain visible.

The new release is pinned by these batch manifests. The global `CURRENT_RELEASE.json`, old releases, original target queue state and completed pilot products are retained. The pilot control directory name is preserved to reuse its existing completion record; all subsequent control cells use the same independent control namespace.

## Preparation and verification

This delivery prepares files only. No pending target or control cell is launched on Mac. Verification uses controller fixtures, configuration and inventory inspection, source/output-presence checks, and saved pilot data for completed-only summary publication. This is not native Windows execution verification and introduces no numerical or pixel parity gate.
