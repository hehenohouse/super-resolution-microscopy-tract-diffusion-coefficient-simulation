# Common protein analysis pipeline

One shared, frozen implementation serves TRAK1, TRAK2, Mac and Windows. Dataset identity, channels, acquisition units and scientific settings live in `configs/`. Existing scientific results and legacy releases remain in their original dataset folders.

This package does not start a batch when installed. It uses the four existing Python environments and the existing Fiji/Java installation; it installs no dependencies and performs no Mac/Windows numerical or pixel replay.

## Speed settings — 2026-11-06

The pending Windows batch is pinned to `releases/20261106T_pending_and_Halo_controls_v5_roicheck`. It builds on `..._v2_fastfit` (faster fit and detection), `..._v3_nellie` (concurrent Nellie) and `..._v4_cleanexit` (running fit workers and Nellie chunks stop promptly when a stage fails). v5 also fixes an assignment self-check that used the pixel ROI instead of the fit's ROI polygon and stopped `WT_TRAK1_BR5_T5`; see `CHANGELOG.md`. These releases only change how work is scheduled and stopped. Every scientific method, parameter and cache identity is unchanged, so completed cells and committed checkpoints are reused.

Before this change, a Windows cell took about 61 minutes. Fit took about 43 of them, with 4 workers and a per-frame wait. Single-process Nellie (about 23 minutes) was hidden behind fit, and TrackMate used 2 threads with a JVM restart every 250 frames. The expected time is now about 20 minutes per cell. That figure is estimated, not yet measured on Windows.

| Setting (`execution` in the batch configs) | Value | Effect | Override |
| --- | --- | --- | --- |
| `fit_workers` | `"auto"` = logical CPUs − 2, at most 10 | Each frame is fitted by one worker, two 25-frame chunks are queued ahead, and fit no longer waits for Nellie | `TRAK_FIT_WORKERS` |
| `detection_threads` | `"auto"` = logical CPUs − 2, at most 12 | TrackMate threads; still one thread per frame | `TRAK_DETECTION_THREADS` |
| `nellie_chunk_workers` | `3` | Independent 25-frame native Nellie chunks run concurrently | `TRAK_NELLIE_WORKERS` |
| TrackMate JVM (code constant) | 625 frames / 6 GiB heap | 3 JVM restarts per cell instead of 9 | — |

Without these keys the code keeps the old behaviour: 4 fit workers, 2 TrackMate threads and 1 Nellie chunk at a time. To confirm the settings at run time, look for these log lines: `"fit_scheduler", "workers": 10`, `"Nellie broad scheduler", "chunk_workers": 3`, `TrackMate threads 12` and `maximum 625 frames`.

If TrackMate ever runs out of Java heap, the batch stops as an infrastructure error and keeps its committed chunks. Setting `TRAK_DETECTION_THREADS=2` reduces native memory use.

To roll back to `..._v1`, repin the three batch manifests and set `fit_workers` back to `4`; older code cannot read `"auto"`. To roll back between v2–v5, only the manifests need to change; before v5, `WT_TRAK1_BR5_T5` fails again in assignment.

Verification was same-machine parity on Mac. Fit tables (decompressed) and Nellie masks, thresholds and indexes were byte-identical to the previous release, with serial and concurrent settings alike. See `CHANGELOG.md` for the details. Native Windows execution of these releases has not been verified yet.

## Combined Windows entry and separate HaloTag-only batch — 2026-11-04

Double-click **`START_ALL_PENDING_WINDOWS.cmd`** in My Book's root. This resumes the existing TRAK1/TRAK2 BR4/BR5 queue, then processes the independent HaloTag-only control queue. At preparation, three target-extension cells and one control pilot were complete; 153 target-extension and 19 control cells remain. Completed cells and committed stages are reused. No installation, new pilot or cross-system qualification is scheduled.

HaloTag-only contains 20 WT cells across two BRs. Its figures, numerical results, On percentage, ROI mask areas, per-cell localization counts, Intra summaries and `Cell_Metrics.xlsx` stay in the negative-control dataset. They are **not merged with TRAK1 or TRAK2**. WT-only summaries use the declared roster and original `NegativeControl/WT/` folder; no TDK or MDK control groups are invented.

Use **`STOP_AFTER_CURRENT_CELL_ALL_WINDOWS.cmd`** to stop the whole combined queue after the current cell, and **`RESUME_ALL_PENDING_WINDOWS.cmd`** to continue. Original batch launchers, paused state, historical results and global release selection remain available. The new batch is pinned to its own frozen release.

See [Windows start, outputs and averaging guide](docs/WINDOWS_ALL_PENDING_AND_HALOTAG.md). This delivery prepares launchers and publishes progress summaries from the completed control pilot; it does not automatically start pending cells.

## BR4/BR5 extension batch — Windows first (2026-11-02)

The new batch prepares **156 new cells** and joins their results with **211
existing completed cells** for five-replica summaries. No cell analysis is
started during preparation. The batch uses its own manifest, frozen release,
state and stop markers; the global `CURRENT_RELEASE.json` and original launchers
remain in place.

| Protein | Existing BR1–BR3 cells | New BR4/BR5 queue | Full five-BR summary |
| --- | ---: | ---: | ---: |
| TRAK1 | 105 | 81 | 186 |
| TRAK2 | 106 | 75 | 181 |
| Total | 211 | 156 | 367 |

On the Windows computer, double-click **`START_NEW_REPLICAS_WINDOWS.cmd`** in
My Book's root or in `_TRAK_pipeline/batches/20261102_BR4_BR5/`. It processes both
proteins using the existing on-drive Windows environments, Fiji and Java. Run
the same file again to continue from committed stages. No package installation,
Mac/Windows parity replay or repeated 150-frame pilot is part of this batch.

`STOP_AFTER_CURRENT_CELL_NEW_REPLICAS_WINDOWS.cmd` requests a clean stop after
the current cell for the entire combined queue. Control-C interrupts the owned
process tree and retains committed checkpoints. Original batch states and stop
requests remain separate. Mac `_MAC.command` counterparts are retained.

New raw files remain at `Xinran_TRAK1/TRAK1/<condition>/<replica>/<cell>` and
`Xinran_FullsetTRAK2/TRAK2/<condition>/<replica>/<cell>`. The user-confirmed TRAK2
MDK mapping is `20260820_MDK_trak2_prcs = BR4` and
`20260910_MDK_TRAK2_prcs = BR5`. Parent replica folders define BR identity;
original `WTbr1`/`WTbr02` filenames are retained. Cross-protein BR pairing remains
unconfirmed and is not inferred from these names.

Production science is unchanged: **raw C1 → per-frame Nellie broad → support
>=10/50 → post-min32 → enclosed holes <64 → ROI**, with the original per-point
**1-SE ellipse** assignment. The isolated static mask from a 2500-frame mean C1
image is an experiment and is not used by this batch. Existing acquisition,
TrackMate, fitting/QC, TARDIS and shared 1000-repeat time-bootstrap rules remain.

Area-normalized Intra summaries use method v2: normalize each eligible cell's
signed curve over the full 0–3 um range, average cells equally within BR, then
average the corrected BR curves equally within condition. Neither mean is
area-normalized again. Signed/peak conventions and common validity rosters are
retained. On% is first calculated per cell as `100*On/(On+Not-on)`, then averaged
equally within BR and equally across available BRs within condition. Unknown and
QC-excluded points remain outside its denominator.

Cell, BR, condition and top-level outputs retain per-cell localization counts,
On%, first-block ROI-mask area (frames 0–49), the time-weighted ROI-mask mean
across the complete movie, SD/SEM and participant counts. Reader workbooks retain
both pooled-cell and equal-BR descriptive count means. Signed/area/peak Intra
figures include full and zoom views plus On/Not-on overlays. PNGs, Excel files
and indexes remain on My Book next to the corresponding cell/BR/condition.
Incomplete or failed members remain explicit; missing data are not filled with
zero. No additional formal significance tests or biological pairing are added.

See [Windows instructions](batches/20261102_BR4_BR5/README_WINDOWS.txt) and the
[batch index](batches/20261102_BR4_BR5/INDEX.html). Preparation validation is
bounded metadata/inventory, logic and synthetic tests. It does not claim a
native Windows scientific run. The manifest is enabled only after its pinned
release has been frozen; this preparation does not start the queue.

## Documentation and source publication — 2026-11-02

**GitHub scope: text source and documentation only.** Java.class, NPZ resources,
Excel templates and deployment release bundles are not included. The live My Book
package retains them. The source archive alone is not a turnkey installation;
provide/rebuild runtime resources before freezing and installing a new copy.



The current source archive includes the shared production pipeline, both-platform launcher examples, existing dependency definitions and opt-in experimental entry points. The detailed [change log](CHANGELOG.md) records both changes and retained behavior; [publication scope](docs/publication_scope.md) explains the code-only archive.

The original completed datasets contain 105 TRAK1 and 106 TRAK2 cells. The BR4/BR5 extension adds 81 TRAK1 and 75 TRAK2 cells; the combined inventory therefore contains 367 cells. The modular migration adopted the original saved results. This documentation/publication update does not re-run those cells. Run, summarize and refresh remain separate operations.

| Production setting | Retained value |
| --- | --- |
| Acquisition | 2500 original frames, 117 nm/pixel, 20 ms/frame |
| Detection | TrackMate 7.14.0 LoG, radius 0.28 um, threshold 6.5, median filter and subpixel localization |
| Fit / uncertainty | Existing 7x7 integrated Gaussian, annular background constraint, full observed-Hessian nominal position SE and original QC |
| Mask | Original raw C1 -> Nellie broad -> support >=10/50 -> FG8 min32 -> BG4 enclosed holes <64 -> ROI |
| Assignment | Original k=1 diagonal SE ellipse, closed-pixel contact including tangency, full mask at frame //50 |
| TARDIS | True original adjacent frames, 0-3 um /300 bins, beta tail 2.1-3 um, retained normalization and zero-distance rules |
| Time bootstrap | 1000 shared paired repeats; primary25 frames,10/50 controls,250-frame strata; no block-join adjacency |
| On fraction | On/(On+Not-on); per-cell values, equal-cell BR and equal-BR condition summaries |

See the [experimental archive](experiments/README.md) for fixed1/2-pixel assignment, kymographs, contrast tests, additive filtering scales and strong-signal supplementation. These remain independent experiments, not production defaults. The full strong-signal TARDIS test covers MDK–TRAK1 / BR1 / T1 only, not WT/TDK or all MDK cells.

## Running from a dataset folder

After launcher installation, open the appropriate file in the dataset's `_batch_pipeline` folder:

| File prefix | Behavior |
| --- | --- |
| `START_BATCH` | Reuse compatible stages, compute missing stages, summarize and publish. |
| `RESUME_BATCH` | Continue after an interruption and clear this modular batch's stop request. |
| `STOP_AFTER_CURRENT_CELL` | Request a clean stop after the current cell. |
| `PLAN_BATCH` | Show reuse, recomputation and missing-input decisions without running science. |
| `REFRESH_FIGURES` or `REFRESH_SUMMARIES` | Render saved scientific products only. Missing science is reported, never computed. |
| `UPDATE_SUMMARIES` | Explicitly compute BR/condition statistics from completed cells, then publish. |

Mac launchers end in `.command`; Windows launchers end in `.cmd`. The existing `UPDATE_CELL_METRICS` and `UPDATE_INTRA_SUMMARIES` shortcuts use the explicit summary operation. Setup and historical validation files are preserved, but normal batch launch does not invoke cross-platform qualification.

Control-C interrupts execution and preserves committed stages/checkpoints. To stop at a cell boundary, use `STOP_AFTER_CURRENT_CELL`. New modular stop markers do not resume or alter historical queues. Keep the complete dataset and common package together on the external drive:

```text
My Book/
├── _TRAK_pipeline/
├── Xinran_TRAK1/
└── Xinran_FullsetTRAK2/
```

All controlled results, state, logs, caches and temporary files remain on the data drive. Installed Python/Fiji/Java executables are reused from their configured locations. If a moved Windows environment cannot start, repair it with the preserved Setup entry point; the shared launcher does not download software silently.

## Computation and cache boundaries

The modules separate detection, refinement/SE, broad segmentation, support counts, thresholding, mask cleanup, ROI intersection, assignment, distance counts, TARDIS decomposition, bootstrap, scientific statistics and presentation.

A scientific artifact is identified by its **explicit method ID/version, canonical scientific parameters and upstream scientific artifact IDs**. Release/source checksums audit code publication; they never invalidate scientific caches because a title or source file changed. Changes to scientific algorithms require a corresponding method-version update.

| Change | Required computation |
| --- | --- |
| Support threshold | Reuse broad masks and counts; redo threshold and affected downstream stages. |
| Support window, participating frames or count rule | Recount support and redo affected downstream stages. |
| Foreground/hole cleanup | Reuse support selection; redo cleanup and affected downstream stages. |
| ROI | Preserve full-field Nellie; invalidate ROI-dependent fitting/selection and downstream products. |
| Spatial assignment | Preserve detection, fit and masks; redo labels and affected group counts/analysis. |
| Labels, selected coordinates, τ or bin edges | Recompute affected distance counts. |
| Decomposition parameters | Reuse compatible raw distance counts. |
| Bootstrap resampling rule | Recompute the affected resampling products with retained frame-anchor units. |
| Colors, fonts, layout or column order | Republish only. |
| Filtering, normalization or averaging | Explicit scientific/statistical computation, followed by publication. |

The v1 distance method supports **τ = 1 original frame only**. Changing a configuration to another τ is rejected until a new implementation and method version are supplied. Resampling never creates adjacent frame pairs across sampled block joins.

## Area-normalized Intra aggregation — summary method v2

Each eligible cell is normalized **before** the BR mean is calculated. Let
$I_i(r)$ be cell $i$'s signed Intra density and
$A_i=\sum_k I_i(r_k)\Delta r_k$ its signed area over the full 0–3 um grid:

$$
g_i(r)=I_i(r)/A_i,\qquad
M_{\mathrm{BR}}(r)=\frac{1}{N_{\mathrm{cells}}}\sum_i g_i(r).
$$

The condition area curve is the equal-BR mean of those corrected BR area curves:

$$
M_{\mathrm{condition}}(r)=\frac{1}{N_{\mathrm{BR}}}
\sum_b M_{\mathrm{BR},b}(r).
$$

There is **no further area normalization** of either mean: averaging unit-area
curves already preserves unit area. BRs with different cell counts still receive
equal weight at condition level. The BR thick area line is the arithmetic mean
of its individually normalized thin cell lines.

This replaces v1's mean(signed Intra)/area(mean(signed Intra)) for the area view.
When cell signed areas are positive, that old expression implicitly weighted
normalized cell shapes by their signed areas. It was not an equal-cell shape mean.
The correction changes scientific hierarchy statistics, not just a legend.

The hierarchy settings are method_version="2",
area_order="normalize_cells_then_equal_cell_mean" and
condition_area_order="equal_BR_mean_of_corrected_area_means". They require new
compatible summary identities; v1 area-summary caches cannot substitute for v2.

Signed means and existing peak-normalization rules are unchanged. Eligibility,
the shared On/Not-on/all-view roster, negative bins, unavailable values,
between-BR sample SD/SEM and original cell results are retained. This does not
refit localizations, alter masks/assignments, recalculate single-cell TARDIS,
change On%, or modify first/mean ROI mask area. Update hierarchy science
explicitly from completed cell products, then publish; pure refresh still
performs no scientific computation. An old or missing hierarchy method in a saved
manifest is reported as incompatible by refresh; it is not relabeled as v2 or
silently recalculated.

For the completed TRAK1/TRAK2 datasets, the targeted entrypoint updates only the
Intra hierarchy from saved cell curves. From this package directory, use the
existing science Python environment:

```sh
python -B tools/update_intra_equal_cell.py --config configs/TRAK1.json --completed-only
python -B tools/update_intra_equal_cell.py --config configs/TRAK2.json --completed-only
```

The default mode explicitly calculates the 13 Intra nodes per protein (9 BRs,
3 conditions and 1 top-level comparison), then publishes full/zoom figures and
On/Not-on overlays. `--compute-only` persists the corrected hierarchy statistics
without plotting; `--publish-only` reads those saved v2 products and only
publishes. Neither mode calls upstream cell science or updates scalar metrics
or Excel workbooks. Earlier v1 data directories remain as historical results.
A new frozen release must be activated before the normal launchers use the
correction; the targeted command does not itself activate a release.

## Scientific conventions retained

- Default grouping uses the original per-localization k=1 diagonal SE ellipse and contact with unions of closed pixel squares, including tangency. It is spatial compatibility, not binding.
- Full-image masks carry explicit TYX coordinates, integer pixel centers, scale, crop offset, original-frame mapping and C1/C2 identity-registration status. Registration remains independently unverified.
- Missing frames or incomplete support blocks yield unknown. Valid empty masks remain valid negatives. Unknown C1 zero-valued pixels and the original FOV-boundary rule remain separate from Not-on.
- Detection UIDs are independent of sorting, row numbers, classification and refined coordinates. Existing fit fields and QC are read-only; new methods write UID-keyed label tables.
- On% is `100 × On / (On + Not-on)`. Unknown and QC-excluded counts remain separate. Total selected localizations are On + Not-on + unknown.
- ROI mask area is reported for the first support block and as the full-movie time-weighted mean. Missing required mask frames make the full-movie mean unavailable; an invalid first block makes the first area unavailable. The first block is not an instantaneous first-frame boundary.
- BR statistics use equal cell weighting. Condition scientific summaries retain equal-BR weighting; localization-count summaries retain both pooled per-cell and equal-replica means. No sum-over-cells count is substituted for a per-cell mean.
- Nominal camera calibration, exploratory uncertainty intervals and the existing signed-curve conventions remain labeled. No cross-cell or genotype significance test is added.

## New proteins

Copy `configs/OTHER_PROTEIN.example.json` to a new named configuration and provide its data root, molecule/sample identity, channels, acquisition metadata, sample discovery/inventory and existing runtime profiles. Keep detection, fitting, mask and TARDIS parameters explicit. Run `plan` before any new computation. A different mask method can replace the mask producer while retaining its coordinate/time contract; it does not require rewriting localization or presentation.

The example has an empty sample list and no runtime profile deliberately. Complete those fields before using it. No new environment is created automatically.

## Developer interfaces

Use an existing science Python executable with `tools/launch.py`:

```text
launch.py plan --config configs/TRAK2.json --cell WT_TRAK2_BR1_T5
launch.py run --config configs/TRAK2.json --cell WT_TRAK2_BR1_T5 --target mask_cleanup
launch.py summarize --config configs/TRAK2.json --completed-only
launch.py refresh --config configs/TRAK2.json --completed-only
launch.py stop --config configs/TRAK2.json
```

The launch script selects `CURRENT_RELEASE.json` and verifies the small frozen-code manifest. Dataset configuration remains outside the frozen release. A missing pointer selects development source explicitly; the installer requires an active frozen release.

Release procedure, using the existing science Python:

1. Run the small test suite and read-only T5 checks; do not start a production batch.
2. Run `tools/freeze_release.py --dry-run`, then create a new unique release using `--release-id <ID> --activate`.
3. Run `tools/launch.py --check-release`.
4. Preview `tools/install_launchers.py --dry-run`; use `--apply` only after the release is ready.

Frozen releases cannot be overwritten. Launcher installation backs up existing launchers before replacing them, and records each change in `<control>/modular/launcher_installation.json`. To roll back, activate a prior common release or restore the exact legacy launcher files from that recorded backup; newly introduced shortcuts are listed in the backup manifest. Original Setup programs, science data and old releases are not replaced.

See `docs/architecture.md` for the stage/data interfaces and migration guarantees. Validation reports are stored in `validation/` on this drive. Passing Mac tests does not claim native Windows execution.

## Historical figures and missing saved products

Complete historical figures are reused without scientific work. Some historical
plot caches contain fewer prepared products than the new 38-image publisher.
If a pure refresh reports missing scientific products, it stops that publication
and reports the missing inputs; it does not compute them. Prepare those products
with an explicit scientific command, then refresh. For example:

```text
python -B tools/launch.py run --config configs/TRAK2.json --cell WT_TRAK2_BR1_T5 --target statistics --force-stage statistics --skip-summary
python -B tools/launch.py refresh --config configs/TRAK2.json --cell WT_TRAK2_BR1_T5 --completed-only
```

This explicitly rebuilds saved plot statistics only; compatible localization,
mask, distance, decomposition, and bootstrap products are reused. No such
preparation is automatically performed by refresh.


Completed on 2026-11-02: six focused regression tests passed; corrected saved-curve hierarchies were regenerated for all 105 TRAK1 and 106 TRAK2 cells (26 nodes, 312 PNGs). Prior cohorts and every signed/peak saved row were unchanged. Upstream cell-science calls, new cells and scalar/Excel rewrites were zero. V1 numeric result directories are retained.


BR4/BR5 preparation checks completed on2026-11-02:37focused tests passed (7inventory,4legacy-source,14summary/normalization,12batch-control). Two real committed legacy-cell adapters read successfully under the isolated new configs. These were code/metadata checks only:0newcells and0scientific stages started. The new queue is pinned to its own frozen release20261102T_BR4_BR5_batch_v1; the prior global release pointer is retained.


Windows Python3.11 startup hotfix (2026-11-02): the BR4/BR5 controller and standalone Intra updater now use os.path.ismount. This replaces unsupported Windows pathlib.Path.is_mount calls. Seventeen focused regressions passed, including simulated unsupported Path.is_mount, volume failure, queue startup, pause/resume and missing-output repair. The batch is pinned to20261102T_BR4_BR5_batch_v2_winmount; original v1/global release and all scientific settings/outputs are retained. No cell analysis was launched by the fix.


BR4/BR5 I/O hotfix2026-11-02: pinned version20261102T_BR4_BR5_batch_v3_iofix uses short on-drive Nellie scratch and supports flat legacy/nested modern scalar metrics. Sixpath and21schema/five-BR tests passed; all211oldcell adapters loaded, and one isolated25-frame native I/O probe succeeded. Committed caches, scientific parameters and cell-first area means remain unchanged. No full batch or cross-system pixel qualification was run.


2026-11-04 detector-memory repair: pinned version20261104T_BR4_BR5_batch_v4_winrun restores maximum250frames per JVM, explicit image flush/GC, checkpoint-preserving resume and heap-exhaustion infrastructure stop. Detector settings and scientific IDs stay unchanged. Seven targeted tests and real625-frame checkpoint reads passed. A native Mac probe could not start detection because existing dependency files were incomplete; no native scientific validation is claimed. Preserve3900detection and4494Nellie chunks; use the same START file to retry failed rows. (Superseded 2026-11-06: 625 frames per JVM with a 6 GiB heap, same frames-per-GiB ratio; see Speed settings.)

The BR4/BR5 manifest enables a first-complete-cell gate for this release. On Windows, the first pending cell must finish its saved scientific outputs and figure publication; success continues automatically, failure stops the combined queue immediately with checkpoints retained. The gate does not run on this Mac and does not claim Windows execution success. No cross-platform replay or pilot was added.


BR4/BR5 C1 display correction (2026-11-04): extension configs use400–2300 ADU for C1 and retain406–554 ADU for C2. This changes presentation identity only; all scientific artifacts and paused batch state remain. The original BR1–BR3 configuration is retained. See the change log and validation/C1_display_update_20261104.


Adaptive C1 display (2026-11-04): future BR4/BR5 publication uses `style.C1_display_mode="roi_percentile"` and `C1_display_percentiles=[1,99.9]`. One per-cell window is derived from finite nonzero ROI pixels in the fixed ten representative frames, then shared by raw C1 panels and the saved C1 mean overlay. Actual limits and fallback reasons are published. This is presentation only; completed fixed-range pictures and all scientific outputs remain. Fixed mode is still supported.
