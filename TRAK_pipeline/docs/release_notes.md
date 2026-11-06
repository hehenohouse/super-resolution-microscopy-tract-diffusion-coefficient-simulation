# Execution speedups v2–v4 and assignment ROI check v5 — 2026-11-06

The pending Windows batch (`batches/20261104_all_pending`, together with its
BR4/BR5 and HaloTag-only child queues) is pinned to
`releases/20261106T_pending_and_Halo_controls_v5_roicheck`. Its predecessors are `..._v4_cleanexit`,
`..._v3_nellie`, `..._v2_fastfit` and `..._v1`.

- v2 changes fit scheduling. Each frame is one worker task, two 25-frame chunks
  are queued ahead, and there are `"auto"` workers (at most 10). Fit no longer
  waits for Nellie. TrackMate uses `"auto"` threads (at most 12) and restarts its
  JVM every 625 frames with a 6 GiB heap, instead of every 250 frames with 2 GiB.
- v3 runs 3 independent native Nellie chunks at once and still commits them in
  chunk order.
- v4 changes only the error path. When a stage fails or is interrupted, running fit
  workers and Nellie chunk processes are stopped at once instead of finishing their
  current frame or chunk. Nothing is committed from a stopped worker or chunk.
- v5 fixes the assignment self-check on sampled frames. It now tests selected
  centers against the same ROI polygon that fit QC used, not the pixel-center
  raster. The mismatch had stopped `WT_TRAK1_BR5_T5` (one boundary point in
  frame 0). Labels of completed cells are unchanged.

Scientific methods, parameters, cache identities, queue membership and
HaloTag-only separation are unchanged. Completed cells and checkpoints are
reused. Before these releases, a Windows cell took about 61 minutes; the
estimate is now about 20 minutes, which has not yet been measured on Windows.
Mac same-machine parity was byte-identical for decompressed fit tables and for
Nellie masks, thresholds and indexes. The suite of 171 tests keeps 9
pre-existing Mac-environment failures: 5 need the cleanup environment's
shapely (all 18 pass with it), and 4 come from the macOS `/var` temporary
symlink. Settings, overrides and rollback are listed in the README section
"Speed settings — 2026-11-06".

# Windows BR4/BR5 extension preparation — 2026-11-02

A separate batch manifest queues only 156 new cells: TRAK1 81 and TRAK2 75.
Completed-only summaries combine them with the 211 original cells, giving
186 TRAK1 and 181 TRAK2 cells (367 total) across five declared replicas per condition.

Windows is the primary run target. Double-click START_NEW_REPLICAS_WINDOWS.cmd
at My Book's root or in the batch folder. The existing on-drive Python/Fiji/Java
installation is reused, with automatic drive-relative path resolution. Running
Start again resumes committed stages; Stop-after-current-cell applies to the
whole combined queue. Mac counterparts and original launchers remain available.
The batch uses a pinned release and isolated state, preserving the global
CURRENT_RELEASE pointer and historical queues.

The new batch retains raw-C1 support50>=10/post-min32/holes<64 segmentation and
original1SE assignment. The static2500-frame-mean C1 mask test is not a production
default. Area-normalized hierarchy curves use cell-first method v2 and equal-BR
condition averaging without final renormalization. On% uses per-cell On/(On+Not-on)
followed by equal-cell BR and equal-BR condition averaging. Existing per-cell
counts, first/time-mean ROI areas, SD/SEM, PNG/Excel/index outputs and signed/peak
full/zoom overlays remain included.

Preparation checked unique inputs and declared ImageJ TIFF metadata for all
156 cells; seven bounded inventory tests and four legacy-control tests passed.
Further integration checks are recorded separately. No scientific cell was
started, no native Windows scientific run is claimed, and no installation,
Mac/Windows numerical/pixel replay or repeated pilot was added. Missing members
remain explicit and cross-protein BR pairing remains unconfirmed.

The manifest is enabled only after its immutable release is present. This
preparation does not launch the queue. See the batch README_WINDOWS.txt and
INDEX.html for the operational entry points and output locations.

---

# Cell-first area-normalization correction — 2026-11-02

Hierarchy summary method v2 changes the area view to **normalize each eligible
cell first, then average cells equally within BR; average those corrected BR
area curves equally within condition**. Neither mean is area-normalized again.
This replaces v1's normalization of the raw BR signed mean.

The new rules are area_order="normalize_cells_then_equal_cell_mean" and
condition_area_order="equal_BR_mean_of_corrected_area_means", with hierarchy
method version 2. Summary identities change; compatible cell fits, masks,
assignments and single-cell TARDIS remain reusable.

Only affected hierarchy scientific summaries and their figures require update.
Signed/peak rules, eligibility rosters, negative/unavailable values, between-BR
SD/SEM, On%, mask areas and per-cell counts retain their definitions. Regression
and regeneration completion status are recorded separately after execution.
See the [change log](../CHANGELOG.md) and [aggregation definition](../README.md).

The targeted tools/update_intra_equal_cell.py command accepts --config and
--completed-only for TRAK1 or TRAK2. It explicitly computes 13 Intra-only nodes
per protein from completed curves and publishes their figures; --compute-only
and --publish-only separate these operations. It does not update scalar
metrics/Excel or run upstream cell science. V1 directories remain available.
Normal launchers receive the correction through a separately activated frozen
release; activation and regeneration status are recorded after execution.

GitHub remains a text-only source/documentation archive. Binary runtime assets,
templates and science data/results remain on My Book. This correction does not
silently start a production batch or add a cross-platform numerical gate.

---

# Source documentation update — 2026-11-02

This update documents the production pipeline and historical repairs, and
separates isolated mask/assignment experiments from batch defaults. See the
[detailed change log](../CHANGELOG.md) for changed/unchanged behavior, retained
scientific settings, acceptance scope and experimental limitations.

The final publication scope is text source and English documentation only;
clean templates and fixed resampling assets remain on My Book. Raw images, ROIs, scientific result tables,
masks, figures, runtime environments, caches and machine-specific inventories
remain on the external drive. No scientific cell or hierarchy analysis is
rerun by this documentation/publication operation.

The active frozen release and existing 211-cell result history remain in place.
The earlier held queue and stored T5 experimental state are not resumed or
rewritten. Existing Mac and Windows launch methods are preserved. No new native
Windows run or cross-platform numerical/pixel qualification is claimed.

The October 30 MDK–TRAK1 / BR1 / T1 strong-signal-union comparison used the
original 1-SE assignment and all 2500 frames. It is a one-cell sensitivity
experiment, not a replacement production mask. Similar normalized Intra curves
do not establish segmentation accuracy. Full numeric and figure products remain
on My Book; this source publication does not export those result files.

---

# Modular v1 release — 2026-10-30

One shared implementation supports TRAK1 and TRAK2 using explicit scientific
method versions and stage-level upstream identities. Historical results remain
in place; adapters read them without migrating or duplicating localization tables.

The release separates full-image Nellie, support counts, thresholding, cleanup,
ROI intersection, assignment, per-frame distances, decomposition, bootstrap,
scientific statistics, and presentation. A saved C1 arithmetic mean is independent.
Refresh cannot invoke scientific stages, including when products are missing.

Completed-only hierarchy aggregation uses the full expected inventory for progress.
Unchanged inputs reuse the committed summary; unchanged nodes skip data and figure
exports. Scientific configuration changes do not mix incompatible cell results.

Validation: 54 bounded fixtures passed. Metadata adoption covered all 211 existing
cells. T5 passed the fixed ten-frame identity/geometry audit; its 22 saved curves
and three shared bootstrap plans reproduced from retained counts. A real dummy
nested-process interruption left no running child. No production science was
recomputed and no new batch was started. Windows launcher quoting was checked
statically; no native Windows numerical or pixel qualification was introduced.

Reports remain in the common package validation directory. Existing launchers
are backed up before installation; original Setup files and legacy releases remain.


Completed on 2026-11-02: six focused regression tests passed; corrected saved-curve hierarchies were regenerated for all 105 TRAK1 and 106 TRAK2 cells (26 nodes, 312 PNGs). Prior cohorts and every signed/peak saved row were unchanged. Upstream cell-science calls, new cells and scalar/Excel rewrites were zero. V1 numeric result directories are retained.


BR4/BR5 preparation checks completed on2026-11-02:37focused tests passed (7inventory,4legacy-source,14summary/normalization,12batch-control). Two real committed legacy-cell adapters read successfully under the isolated new configs. These were code/metadata checks only:0newcells and0scientific stages started. The new queue is pinned to its own frozen release20261102T_BR4_BR5_batch_v1; the prior global release pointer is retained.


Windows Python3.11 startup hotfix (2026-11-02): the BR4/BR5 controller and standalone Intra updater now use os.path.ismount. This replaces unsupported Windows pathlib.Path.is_mount calls. Seventeen focused regressions passed, including simulated unsupported Path.is_mount, volume failure, queue startup, pause/resume and missing-output repair. The batch is pinned to20261102T_BR4_BR5_batch_v2_winmount; original v1/global release and all scientific settings/outputs are retained. No cell analysis was launched by the fix.


BR4/BR5 I/O hotfix2026-11-02: pinned version20261102T_BR4_BR5_batch_v3_iofix uses short on-drive Nellie scratch and supports flat legacy/nested modern scalar metrics. Sixpath and21schema/five-BR tests passed; all211oldcell adapters loaded, and one isolated25-frame native I/O probe succeeded. Committed caches, scientific parameters and cell-first area means remain unchanged. No full batch or cross-system pixel qualification was run.


2026-11-04 detector-memory repair: pinned version20261104T_BR4_BR5_batch_v4_winrun restores maximum250frames per JVM, explicit image flush/GC, checkpoint-preserving resume and heap-exhaustion infrastructure stop. Detector settings and scientific IDs stay unchanged. Seven targeted tests and real625-frame checkpoint reads passed. A native Mac probe could not start detection because existing dependency files were incomplete; no native scientific validation is claimed. Preserve3900detection and4494Nellie chunks; use the same START file to retry failed rows.

The BR4/BR5 manifest enables a first-complete-cell gate for this release. On Windows, the first pending cell must finish its saved scientific outputs and figure publication; success continues automatically, failure stops the combined queue immediately with checkpoints retained. The gate does not run on this Mac and does not claim Windows execution success. No cross-platform replay or pilot was added.
