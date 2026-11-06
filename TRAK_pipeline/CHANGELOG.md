# Change log

This log separates production changes, reporting repairs and isolated experiments.
Historical dates describe recorded work, not new analysis during publication.
The November 2 update prepares English source documentation for GitHub; scientific data remain on My Book.

## 2026-11-06 — Assignment selection check uses the fit ROI polygon (release 20261106T_pending_and_Halo_controls_v5_roicheck)

Fixes the only scientific cell failure in the pending batch so far: `WT_TRAK1_BR5_T5` stopped in assignment on 2026-11-05 with `Selected refined center lies outside supplied ROI`.

### Cause

Fit QC selects points with the continuous ROI polygon (`fitted_center_inside_ROI`, `locbio.points_in_polygon`). The assignment self-check on the sampled frames re-tested selected centers against the pixel-center ROI raster. A refined center at (205.414, 144.939) in frame 0 is inside the polygon but lies in pixel (205, 145), whose center is outside it (raster 0). The two definitions disagreed and the whole cell was stopped. In that cell, 2 of 220,001 selected points are in this boundary situation; only the frame-0 point fell on a sampled check frame.

### Changed

- `stages/assignment.py`: the sampled-frame selection check now uses the same frozen polygon function and the ROI vertices that fit QC used. A selected center outside the polygon still raises the same error. If an ROI artifact has no vertices, the previous raster check is used. Classification, the independent pixel-enumeration check, parameters and cache identities are unchanged.

### Verification

- Mac, real inputs read only, outputs in scratch: `WT_TRAK1_BR5_T5` assignment fails with the previous release exactly as on Windows and completes with this release (On 120,244; Not-on 99,757; unknown 0; selected 220,001; QC-excluded 26,060). The frame-0 boundary point is labelled non-mito-compatible by the unchanged classifier.
- Completed cell `TDK_TRAK1_BR4_T10`: all 25 assignment output files (decompressed) are byte-identical between the previous release and this release. Labels and count tables also match the Windows-saved outputs byte for byte; the summary JSON matches field by field.
- New tests (`test_assignment_roi_polygon_check.py`, cleanup environment): a polygon-inside / raster-outside boundary center is accepted, a center outside the polygon still stops the cell, and the raster fallback is kept when there is no polygon. Cleanup environment: 22 assignment tests pass. Science environment: 177 tests, with the 9 pre-existing Mac-environment failures plus these 4 new shapely-dependent tests.
- On resume, `WT_TRAK1_BR5_T5` reuses its cached upstream stages and recomputes only assignment and later stages; its partial assignment folder is empty. No native Windows execution is claimed.

## 2026-11-06 — Prompt cleanup on stage errors (release 20261106T_pending_and_Halo_controls_v4_cleanexit)

Follows an independent read-only review of v3. Results and scheduling are unchanged; only error-path cleanup differs.

### Changed

- Fit (`stages/localization.py`): the process pool is no longer closed twice. On any error or interrupt, `stop_process_pool` cancels queued frames, terminates the running fit workers and joins them; on success the pool is shut down once. Workers only return results, so nothing is committed from a terminated worker.
- Nellie (`stages/mask.py`): native chunk processes are started with `Popen` and tracked. On any error or interrupt the stage marks itself stopping (no new chunk starts), cancels queued chunks and terminates running chunk processes (terminate, then kill after 8 s), instead of letting them run to the end of their chunk. A stopped chunk has no completion marker; on resume its scratch is replaced and it is recomputed. A failing chunk still keeps its scratch and reports its native exit code.
- Stops requested from outside the stage are unchanged: the parent still ends the whole stage process tree (Windows job object after an 8 s grace period).

### Verification

- Same-machine parity after the change (Mac): fit with 8 workers (frames 0–49) and Nellie with 3 chunk workers (frames 0–99) remain byte-identical to the v1 release outputs (decompressed fit tables; packed masks, thresholds and index).
- New tests: a real spawn pool with sleeping tasks is stopped in well under a second with all workers gone and no task completing normally; a failing Nellie chunk terminates the other running chunks, starts no further chunk and commits nothing. Two existing native-path tests now drive the same `subprocess.run`-style fakes through a `Popen` adapter; their assertions are unchanged.
- 173 tests; the same 9 pre-existing Mac-environment failures (5 pass under the cleanup environment, 4 are macOS `/var` temporary-symlink artifacts).
- No native Windows execution is claimed.

## 2026-11-06 — Concurrent Nellie broad chunks (release 20261106T_pending_and_Halo_controls_v3_nellie)

Motivation: after the v2 fit speedup, single-process Nellie (about 14 s per 25-frame chunk, about 23 minutes per 2500-frame cell on Windows) became the per-cell critical path that support counts and every later stage wait for.

### Changed

- `stages/mask.py` `execute_broad`: each 25-frame chunk is still one independent native Nellie process with its own input, scratch and outputs; up to `execution.nellie_chunk_workers` of them now run at once. Chunk results, sampled-frame checks, `chunk_*.json` completion markers, records and checkpoint progress are committed strictly in chunk order. Committed chunks are reused unchanged. Default without the key remains 1; `TRAK_NELLIE_WORKERS` overrides. The three batch configs set 3.

### Verification

- Same-machine parity (Mac, TDK_TRAK1_BR4_T10 C1 frames 0–99, four chunks): previous release, this release with 1 worker and with 3 workers give identical packed masks, frame IDs, `thresholds.json`, `actual_parameters.json` and `broad_index.json`. 100 frames: 17.1 s with 1 worker, 8.8 s with 3.
- Test suite: 171 tests; three new tests (out-of-order completion versus serial output, resume of committed chunks, worker-count rules) pass, and the same 9 pre-existing Mac failures remain. Running the shapely-dependent mask/assignment tests with the cleanup environment passes all 18, so 5 of the 9 are environment-only; the other 4 come from the macOS `/var` → `/private/var` temporary-directory symlink.
- No native Windows execution is claimed by this entry.

### Retained

- Nellie version, filter radii, call sequence, per-chunk inputs and every scientific parameter and cache identity; all v2 changes and everything listed as retained there.

### Documentation

- README "Speed settings — 2026-11-06" (settings, overrides, log lines, rollback), `docs/release_notes.md`, the Windows guide (`docs/WINDOWS_ALL_PENDING_AND_HALOTAG.md` = `batches/20261104_all_pending/README.txt`) and the drive-root `READ_ME_ALL_PENDING_WINDOWS.txt` describe both speedups. The README's 2026-11-04 detector note is marked as superseded. These documentation edits were made after the v3 freeze; the frozen v3 copy keeps the earlier text.

## 2026-11-06 — Execution-only speedup for localization (release 20261106T_pending_and_Halo_controls_v2_fastfit)

Motivation: on the Windows workstation (Xeon Silver 4214R, 12 cores/24 threads) the fit stage took a median 43 of the 61 minutes per cell with 4 workers and a per-frame barrier; fit also waited for Nellie although it does not depend on it, and TrackMate used 2 threads with a JVM restart every 250 frames.

### Changed

- Fit scheduling (`stages/localization.py`): one task per original frame; the frozen annular background and the per-candidate `fit_one` calls now run inside the worker; up to two 25-frame checkpoint chunks are in flight; chunks still commit strictly in frame order with the same marker/table format. Each task carries the frame's detections and raw uint16 pixels; ROI mask, vertices and fit settings are sent once per worker. The previous code sent raw plus a float64 converted frame with every per-frame worker piece.
- `execution.fit_workers` and new `execution.detection_threads` accept an integer or `"auto"` (logical CPUs minus 2, capped at 10 fit workers / 12 TrackMate threads). `TRAK_FIT_WORKERS` and `TRAK_DETECTION_THREADS` override them. Defaults without a config key remain 4 and 2.
- Runner (`runner.py`): the background Nellie broad stage is awaited only by stages that depend on it, so detection-dependent fit no longer waits for Nellie.
- TrackMate (`run_detection.detect`): thread count is a parameter (default 2). TrackMate assigns one thread per frame, so per-frame LoG detections are unchanged; only the global native spot-ID counter order may differ, as it already could with 2 threads.
- Native detection JVM: 625 frames per JVM with a 6 GiB heap (was 250 frames / 2 GiB), keeping the frames-per-GiB ratio of the November 4 detector memory fix; 3 restarts per 2500-frame cell instead of 9. Heap exhaustion still stops the batch as an infrastructure error with committed chunks retained.

### Verification

- Same-machine parity (Mac, cell TDK_TRAK1_BR4_T10, 100 and 50 original frames): previous release vs this release with 4 and 8 workers give byte-identical decompressed chunk tables, `localizations.csv.gz` and `per_frame_counts.csv`. Compressed `.gz` bytes differ only by the PID-bearing temporary filename in the gzip header, which also differs between runs of the previous release.
- 50 frames, same code, local inputs: 4 workers 100 s, 8 workers 30 s (busy Mac; indicative only).
- Test suite: 168 tests; the same 9 pre-existing failures as the previous release under the same Mac environment (missing `shapely` in the science environment, two legacy-source fixtures, one mount-check mock). Six new tests pass, including fit/broad overlap and execution-count rules; detection range tests were updated for 625-frame JVM windows.
- No native Windows execution is claimed by this entry.

### Retained

- All scientific method IDs, versions and parameters; `execution` is not part of any scientific cache identity, so completed and partially completed cells and checkpoints are reused unchanged.
- TARDIS, bootstrap, distances, masks, assignment, statistics, presentation, summaries, controllers/launchers, queue membership and HaloTag-only separation are unchanged.
- Previous releases and the global `CURRENT_RELEASE.json` are untouched; batch manifests pin the new release.

## 2026-11-04 — Prepare a combined Windows entry with isolated HaloTag-only summaries

### Added

- Add one drive-root `START_ALL_PENDING_WINDOWS.cmd` entry, plus matching Resume, global Stop After Current Cell, Plan, Update Summaries and Refresh Figures files. Reuse installed Windows science/detector/Nellie/cleanup environments, Fiji and bundled Java. No package download, setup, cross-system numerical/pixel replay or repeated pilot is introduced.
- Add an outer controller that invokes the original target-extension child queue and then a separate control child queue. It preserves the original BR4/BR5 state rather than rebuilding or resetting it. A global stop marker prevents the next child queue from starting; ownership-aware batch locks and owned-process termination retain duplicate-start and interruption behavior.
- Prepare an explicit 20-cell HaloTag-only inventory: WT BR1/BR2, ten cells each. Queue the completed pilot alongside the other cells so valid completed source/method/output records are adopted and skipped. Preserve the pilot control namespace and scientific caches instead of copying or refitting them.
- Keep the target summary inventories at TRAK1=186 and TRAK2=181. Keep the control inventory at 20 in its own dataset. Preparation observes three completed target-extension cells and one completed control cell, leaving 172 pending cells; runtime derives its own current counts.
- Shorten six mask/assignment temporary filename sites using exclusive same-directory short names. The Halo control folder is longer than the existing target folders; old UUID suffixes would extend actual Windows paths to 264–269 characters. Keep formal filenames, directories, atomic replacement, concurrent uniqueness, numerical outputs and scientific cache identities unchanged. Five focused tests cover path budgets, lossless packed masks, JSON/CSV/NPY/gzip round trips, 48 concurrent writes and failure cleanup.
- Extend hierarchy summaries to declared nonempty ordered subsets of WT/TDK/MDK. WT-only control summaries omit undeclared conditions, retain pending expected BRs as unavailable, and write WT outputs to the explicit original `NegativeControl/WT/` directory.
- Keep full WT/TDK/MDK method-v2 parameter identities unchanged. Record subset condition membership in its scientific summary parameters. Explicit output-directory mapping affects source/publication records without invalidating localization or mask caches.
- Include cell/BR/condition/dataset metrics, reader Excel, complete precision numerical summaries, participant counts, Intra full/zoom and On/Not-on overlay publication in the control batch. Main count averages remain per cell. Technical provenance stays outside the reader workbook.
- Add controller tests for completion adoption and changed-source rejection, stop propagation across child boundaries, resume, independent failure handling, missing infrastructure, queue isolation and zero cell execution during plan/summary/refresh. Add summary fixtures for WT-only membership, unavailable BRs, output folders and the existing full-condition identity.

### Retained

- Preserve original raw C1/C2 pixels, ROIs, completed cell results and scientific stage caches, including all existing target BR1–BR3 results and the paused BR4/BR5 queue. No pending cell is run on Mac during preparation.
- Retain raw C1 Nellie broad, support >=10/50, post-min32, holes<64, original per-point 1-SE ellipse classification, original localization/error/QC settings, τ=1 TARDIS, negative bins, sparse-tail unavailability and the shared 1000-repeat time-block bootstrap.
- Retain On%=100*On/(On+Not-on), unknown/QC exclusions, first-support-block ROI mask area and time-weighted whole-movie ROI mask area. BR means weight cells equally; condition means weight available BRs equally. Save pooled-cell and equal-BR count means separately, with SD/SEM/n in numerical data.
- Retain cell-first area normalization before equal-cell and equal-BR Intra averaging. Do not normalize the final mean again. Keep common view/group cohorts, signed and peak definitions and independent BR design.
- Retain adaptive C1 display only as presentation, fixed C2 display, nominal camera calibration, unverified identity registration, all 38 default per-cell PNGs and the original Mac/Windows launch methods.
- Keep HaloTag-only separate from target biological inference. No new hypothesis test, shifted-mask null, contrast preprocessing or pixel-distance classification is added.
- Preserve the global CURRENT_RELEASE selector and all previous frozen releases. The new entry pins its release in batch manifests. Windows compatibility checks here are bounded code/configuration and controller-fixture checks, not a claim of native Windows execution.

## 2026-11-04 — Complete one HaloTag-only negative-control pilot on Mac

- Discover20WT HaloTag-only cells (BR1/BR2,10each); execute only WT_HaloTagOnly_BR1_T1 from2026-09-30.
- Record the correct control design: fluorescently labeled HaloTag without TRAK1/TRAK2 fusion, not fluorophore-only or untransfected cells.
- Use frozen v5 original raw-C1 support50/post-min32/holes<64 masks, original own1SE ellipse, original detection/fitting/QC and TARDIS/bootstrap rules, with per-cell adaptive C1 display.
- Reuse installed Mac packages through isolated on-drive process adapters: system NumPy first, working common packages before role packages, and validated on-drive Python bytecode caches. No installed package/source file was changed and nothing was downloaded.
- Complete the full2500frames and all14scientific stage calls (including ROI rasterization), with exactly one new cell and38PNG figures. Pipeline elapsed approximately17.5minutes.
- Save11301On,24507Not-on,0unknown;35808selected,4083QCexcluded and39891ROIcandidates. On%=31.559986595174262.
- FirstROI support-mask area186.389424um2; time-weighted mean176.44765086um2. All50mask blocks valid.
- Both full-movie Intra curves pass existing validity checks, with1000/1000sharedvalid repeats forL10/L25/L50. Only the first3On and first5Not-on250-frame periods pass sparse-tail checks; preserve unavailable reasons forlaterperiods ratherthan filling zeros.
- Record actual adaptive C1 display417–1233ADU and fixed C2 display406–554ADU. Save all masks, candidate/fit/QC/label tables, distances, curves, bootstrap products, parameters, provenance and English logs on My Book.
- Independent saved-data audit confirms UID/frame/QC correspondence and count conservation. No other19control cells were started; the paused TRAK BR4/BR5 queue and its state are unchanged. No cross-cell comparison or significance inference is performed.

## 2026-11-04 — Adapt C1 display limits per cell for future BR4/BR5 publication

### Changed

- Add a presentation-only C1 display policy, `style.C1_display_mode = "roi_percentile"`, with 1st/99.9th percentiles of finite, nonzero pixels within the original cell ROI.
- Derive one display interval per cell from the fixed ten representative original frames: 0,49,200,625,1200,1250,1875,2000,2450,2499. Keep the same interval across that cell's frame panels and its saved 2500-frame arithmetic-mean C1 overlay.
- Record the requested policy, sampled frames, sample size, resolved ADU limits and any degenerate-input fallback in the presentation metadata and figure index; show actual display limits in C1 figure captions.
- Preserve fixed-range mode for previous configurations and for an explicit no-positive-pixel display fallback. Constant images receive a documented nondegenerate display interval; missing input frames are not filled or interpolated.
- Enable the adaptive policy only in the current TRAK1/TRAK2 BR4/BR5 extension configs. Freeze an independent release for future Windows publication while preserving the existing launch file and the old releases.

### Retained and scope

- This computes a display window only. It does not rescale stored C1 pixels, alter Nellie input, masks, ROI, localization/QC, assignment, On%, TARDIS, bootstrap or hierarchy statistics.
- Preserve the existing 400–2300 ADU T1–T3 galleries and their completion records. Do not regenerate completed cells or resume the paused queue while preparing this change.
- Scientific method families and scientific cache IDs remain unchanged; requested display policy changes presentation identity only.
- Include the sampling-frame list explicitly in display style/cache identity; changing selected display frames invalidates presentation only. Six focused display-policy tests passed, and a read-only T1 check read exactly ten pages and resolved400–1897 ADU. Scientific algorithms/cache/runner/runtime sources are byte-identical to v4; zero scientific stages or new cells were run.
- C2 display remains 406–554 ADU. All frame numbers, coordinates and units remain original.
- Different cells may use different brightness limits, so the published limits identify the display scale; displayed brightness is not an across-cell intensity measurement.
- No cross-system numerical/pixel qualification, pilot, package installation or new cell execution is introduced. Passing local display-policy checks does not claim a native Windows run of the new publisher.

## 2026-11-04 — Correct BR4/BR5 C1 display contrast without scientific recomputation

### Changed

- Change only `style.C1_display_ADU` in the TRAK1 and TRAK2 BR4/BR5 extension configurations, from 407–651 to 400–2300 ADU. Keep C2 at 406–554 ADU.
- Select a common, fixed display interval from the pooled nonzero original-frame-0 ROI pixels of WT_TRAK1_BR4_T1, T2 and T3: 331,435 pixels; 1st percentile 406 ADU and 99.9th percentile 2294 ADU, rounded outward for display.
- Refresh only the ten representative four-panel images and the saved-mean Not-on overlay for each of these three completed cells. Read original TIFF pages and existing scientific caches; copy the other PNGs unchanged.
- Store refreshed pictures in a new presentation-style gallery and retain the previous gallery and its immutable cache commit. Update the cell's presentation reference after successful publication.
- Supply a lean C1-only refresh utility using existing NumPy, Pillow and Matplotlib, with standard CSV I/O. It avoids the damaged local pandas import without installing or modifying dependencies.
- Ignore Mac AppleDouble `._` sidecars when matching PNG rosters. Publication completed for all three cells: 33 C1 images rendered, 27 other PNGs per cell copied unchanged, and 38 real PNGs per gallery. All scientific artifact IDs/file statistics, old galleries and combined paused state were retained; scientific-stage calls were zero.

### Retained and scope

- Raw C1/C2 pixels, saved mean C1, masks, ROI, detections, UIDs, localization fits/errors/QC, assignment labels, counts, TARDIS, bootstrap and hierarchy statistics remain unchanged.
- Scientific method parameters, upstream artifact IDs and method families remain unchanged; only presentation identity includes the changed style.
- No segmentation, fitting, classification, temporal averaging, TARDIS or scientific statistics are run by this display refresh. No intensity rescaling is written into analysis inputs.
- The original BR1–BR3 display configurations, existing Mac/Windows launch methods, pinned Windows release and paused batch state remain unchanged. Later new-cell publication reads the updated external display configuration.
- This is a visibility correction; it does not establish segmentation accuracy. The display interval was sampled from three TRAK1 cells, not calibrated independently for every new cell/protein.

## 2026-11-04 — Restore bounded TrackMate JVM lifetime and preserve partial detection

### Root cause and changes

- The actual Windows batch recorded156failed cells, all with Java heap exhaustion after625detected frames. The modular detector had omitted the prior maximum250frames per JVM and kept one2GiB JVM for the full2500-frame movie.
- Restore fresh detector JVM processes handling at most250original frames each, with unchanged25-frame detection chunks and the same2GiB heap. This is a resource boundary, not a detector-method parameter.
- Close and flush headless ImagePlus pixels after each chunk, drop Python references and request Python/Java garbage collection. Record per-chunk heap usage and explicit process ranges.
- Treat Java OutOfMemoryError/heap exhaustion as infrastructure exit74. Stop the batch immediately and preserve checkpoints instead of trying the same resource failure on every cell.
- Reuse every compatible committed25-frame CSV/JSON verbatim. Skip fully completed250-frame ranges without starting a JVM; partially completed ranges process only missing chunks.
- Preserve existing nativeSpot IDs and chunk source keys. Persistent UID remains sample+detector artifact+chunk source_key+nativeSpot ID; do not renumber saved detections. IDs in new, uncommitted chunks may start in a fresh JVM namespace, which the chunk key already distinguishes.
- Publish20261104T_BR4_BR5_batch_v4_winrun as an independent pinned release; retain earlier releases and original batch state.
- Enable a batch-specific first-complete-cell gate for this release: the first pending cell must finish scientific products and figure publication before the rest of the queue proceeds automatically. A first-cell, source or publication failure saves state and stops the combined queue with a nonzero exit. Successful verification is keyed to the pinned release; a later restart reuses it.
- This gate is a normal full-cell execution on Windows, not a Mac/Windows parity test, replay, repeated pilot or extra dependency installation.

### Verification and retained behavior

- Seven focused lifecycle/resume/UID/resource-error tests passed. Real saved625-frame checkpoints from representative TRAK1/TRAK2 cells are accepted without modification.
- Eighteen stdlib batch-controller tests passed (six new first-cell-gate cases and twelve existing cases). These use injected backends and start zero scientific workers; they do not claim native Windows execution.
- Audit found3900saved detection chunks and4494saved Nellie chunks across156cells; these were not deleted or overwritten by this repair.
- A local native probe was attempted in a private directory, but the existing Mac Python environment contained missing/incomplete dependency files and stopped during import. No native detection success, Windows scientific replay or pixel-equivalence result is claimed.
- Keep TrackMate7.14.0 LoG radius0.28um, threshold6.5, median filtering, subpixel localization, original raw pixels/frames, fit formulas/QC, original1SE assignment, support50mask, TARDIS/bootstrap and cell-first/equal-BR summary rules.
- No environment installation, RAM-size increase, scientific cache identity change or full-batch restart was performed by the repair.

## 2026-11-02 — BR4/BR5 Nellie path and legacy-summary schema fix

### Changed

- Diagnose the actual Nellie failure from chunk_000000_nellie.log: its auto-generated OME filename was278characters, beyond the affected Windows path limit.
- Move only transient native Nellie files into a short owned workspace under the common external-drive package. The tested equivalent Windows filename is145characters.
- Key temporary-work ownership by full artifact, sample and dataset identity; refuse ownership conflicts/escaping paths. Preserve failed scratch and expose the native error log tail.
- Keep committed packed broad masks, checkpoint paths, artifact IDs and all scientific parameters unchanged. Completed native chunks remain resumable.
- Fix the old-cell summary reader: legacy saved scalar metrics are flat, while modern metrics use nested counts. Read both explicit schemas, cross-check authoritative classification counts and retain count conservation.
- Derive per-cell On% from saved On/(On+Not-on), and read saved first/time-weighted ROI areas independently. Missing/inconsistent data remain errors; no inference or upstream recomputation fills them.
- Publish the independent20261102T_BR4_BR5_batch_v3_iofix release; retain earlier releases and global active selection.

### Verification and retained behavior

- Six short-workspace tests and21scalar-schema/five-BR tests passed.
- Read all211existing cells through the actual new batch configuration:105TRAK1 and106TRAK2, zero scalar/Intra adapter errors, no upstream scientific calls.
- Run only the failed25-frame chunk in an isolated native file-I/O probe: Nellie produced threeOME files and a25x684x428boolean mask, with no cross-system pixel comparison or production-cache insertion. Remove the owned probe images after recording success.
- Retain the exact156-cell new queue,367-cell five-BR summaries, original localization/QC/mask/1SE/TARDIS/bootstrap methods, and cell-first/equal-BR normalization v2.
- Retain original records, partial scientific caches and Windows-owned locks. A Windows restart must use the same START_NEW_REPLICAS_WINDOWS.cmd after the previous run is ended.
- Detection was not committed when broad failed; do not report it as completed. No full batch was started by this repair.

## 2026-11-02 — Windows Python 3.11 mount-check startup fix

### Changed

- Fix the new BR4/BR5 controller's pre-start volume check and infrastructure-error check: use os.path.ismount instead of pathlib.Path.is_mount, which raises NotImplementedError on Windows Python 3.11.
- Apply the same portability correction to the standalone equal-cell Intra update tool; normal batch summaries do not invoke that standalone entry.
- Publish an independent patched frozen release and switch only this batch's pinned release after validation. Preserve the failed v1 release for audit.
- Add focused regression coverage that makes Path.is_mount raise, exercises the real plan entry and checks mounted/unmounted infrastructure classification.

### Retained

- The exact 156-cell execution queue and all367-cell five-BR summary inventories.
- Original raw-C1 support50/post-min32/holes<64 mask, original1SE ellipse, localization/QC, TARDIS and bootstrap parameters.
- Cell-first area-normalized means, equal-cell BR weights and equal-BR condition weights, without final renormalization.
- Existing science caches, completed-cell records, stop/resume state, original launchers and global active release pointer.
- No new cell or scientific stage is started by this patch. The reported failure happened before cell execution.

## 2026-11-02 — Prepare the Windows BR4/BR5 extension and five-replica summaries

### Changed

- Register 81 new TRAK1 cells and 75 new TRAK2 cells from the nested protein folders already on My Book.
- Prepare a new-only 156-cell execution queue and full summary inventories of 186 TRAK1 and 181 TRAK2 cells.
- Include original BR1–BR3 completed results as read-only summary references, giving five declared BRs per condition after the new queue completes.
- Keep new raw folders and original cell numbering unchanged; do not rename or copy TIFFs/ROIs.
- Record the user-confirmed TRAK2 MDK mapping: 20260820 = BR4 and 20260910 = BR5.
- Respect parent BR folders even where original WTbr1/WTbr02 filename prefixes differ between proteins; infer no biological pairing.
- Store portable configs, an exact new-sample queue, metadata inventory audit and batch manifest under batches/20261102_BR4_BR5.
- Separate new batch state and stop markers from historical batch state; read original completion and scientific-definition records through an explicit legacy-control reference.
- Provide one Windows start file for both proteins, plus resume, stop-after-current-cell, plan, completed-only summary and saved-figure-refresh launchers.
- Keep Mac counterparts available without replacing original launch methods.
- Pin the new batch to its own immutable common release; do not change the global active release pointer to select this queue.
- Extend hierarchy membership/progress handling to BR1–BR5 and retain normalization method v2.
- Save BR/condition/top summaries over old and newly completed cells, with expected counts that expose unfinished or failed members.

### Scientific behavior retained

- Original raw C1, per-frame Nellie broad, support >=10/50, post-min32, enclosed holes <64 and original ROI.
- Original own-1SE closed-pixel ellipse assignment, original raw-frame unknown rules and original localization/QC selection.
- Acquisition at 2500 frames, 117 nm/pixel and 20 ms/frame; existing detection, fitting, camera assumptions and uncertainty definitions.
- Original adjacent-frame TARDIS, signed bins/validity guards and 1000 shared paired temporal bootstrap repeats.
- Area view: normalize each cell first, equal-cell BR means, then equal-BR condition means, with no final area renormalization.
- Existing signed/peak conventions, shared membership, negative bins and explicit unavailable reasons.
- Cell On% = 100*On/(On+Not-on), followed by equal-cell BR and equal-BR condition averaging.
- Per-cell total/On/Not-on counts, unknown/QC-excluded counts, first support-block ROI area, time-weighted ROI area, sample SD/SEM and participating counts.
- Full/zoom signed, area and peak Intra PNGs, On/Not-on overlays, reader Excel exports and hierarchy indexes.
- Original completed scientific artifacts and histories; the 211 original cells are not part of the new execution queue.
- Nominal camera calibration, independently unverified identity registration, unconfirmed cross-protein BR pairing and descriptive statistical scope.

### Preparation and scope

- The input audit read first TIFF IFDs and explicit ImageJ metadata for all 312 new C1/C2 files: declared 2500x684x428 uint16, 117 nm/pixel and 20 ms/frame.
- The audit checked unique C1/C2/ROI inputs for 156 cells. It did not read image pixels, enumerate every TIFF page or repeat full-file SHA scans.
- Seven bounded inventory/config tests passed. Four legacy-control tests passed separately; remaining integration results are recorded in the batch validation directory when completed.
- No scientific stages or new cells were started by preparation. No native Windows runtime execution is claimed here.
- No installation, cross-system numerical/pixel qualification, repeated 150-frame pilot, static-mean production mask or new formal significance test was added.
- The separate WT TRAK1 BR1 T1 static mean-mask experiment remains an isolated comparison; it is not the new batch mask method.
- Prepared configs remain non-executable until the manifest references an existing pinned release. Activating that manifest is separate from running its queue.

## 2026-11-02 — Correct cell-first area normalization (summary method v2)

### Changed

- Correct the BR area mean to average individually signed-area-normalized cell curves.
- Define each cell shape as signed Intra divided by its own signed area over 0–3 um.
- Give each eligible cell one equal weight within its BR.
- Average corrected BR area means with equal BR weights at condition level.
- Do not pool cells across BRs with different cell counts.
- Do not normalize either mean again: its inputs already have unit signed area.
- Replace the prior BR normalize(mean(raw signed Intra)) operation for the area view.
- With positive cell areas, that old order implicitly gave larger-area cells more shape weight.
- Make the displayed thick BR area line the direct mean of normalized thin cell lines.
- Replace the old “Signed mean, then area-normalized” area legend/caption.
- Version this as hierarchy summary method 2, a scientific averaging correction.
- Record area_order="normalize_cells_then_equal_cell_mean".
- Record condition_area_order="equal_BR_mean_of_corrected_area_means".
- Require distinct v2 hierarchy cache identities rather than reusing v1 area means.
- Rebuild affected hierarchy science from completed cell curves before republishing figures.
- Add tools/update_intra_equal_cell.py --config ... --completed-only for this targeted correction.
- Separate --compute-only aggregation from --publish-only rendering.
- Prepare 9 BR, 3 condition and 1 top comparison Intra node per protein.
- Keep scalar summaries and Excel outside that targeted update.
- Reject old/missing summary-method metadata during pure refresh instead of repainting v1 as v2.
- Retain prior v1 result directories; new active-release publication is a separate step.

### Retained

- Original localization fits, UIDs, QC, position errors and coordinates.
- Masks, frame mapping, 1-SE assignments, group counts and On percentages.
- Per-cell signed Intra, beta, distance counts and temporal-bootstrap products.
- Existing signed-mean and peak-normalization conventions.
- Shared eligibility rosters across On/Not-on and all views, original guards and unavailable reasons.
- Negative bins, full precision and between-BR sample SD/SEM.
- First/time-mean ROI-mask areas and per-cell localization-count definitions.
- Zero upstream cell-science calls for completed-only hierarchy updates.
- Zero scientific calls for pure refresh; missing science still requires explicit summarize.
- Mac/Windows shared implementation, external-drive storage and text-only GitHub scope.

This entry defines the correction. Regeneration, regression-check and publication
completion records are written after those operations; documentation alone does
not claim their completion.

## 2026-11-02 — Source publication and documentation

### Changed

- Consolidated the English history of changed and unchanged behavior.
- Documented the shared TRAK1/TRAK2 package, cache contracts and public commands.
- Documented preserved Mac/Windows launchers and configured existing runtimes.
- Separated production defaults from isolated T5 and MDK sensitivity experiments.
- Prepared source and documentation for publication. Runtime assets were reviewed separately; the final upload is text-only at explicit user direction.
- Kept technical provenance separate from reader-facing metric tables.
- Preserved the earlier release notes below the new publication summary.

### Not changed by this publication

- No production-cell detection, refinement, segmentation, assignment, distance or TARDIS calculation is started.
- No existing biological cell, BR or condition statistics are scientifically recomputed by this documentation operation.
- Raw TIFFs, ROIs, localization masters, masks and cell-result tables are excluded from the source upload.
- Runtime environments, downloaded Python/Fiji/Java software, caches and temporary files are excluded.
- Result-specific bootstrap arrays, galleries, figures and machine inventories remain on My Book.
- Fixed reference resampling assets remain method resources on My Book; NPZ files, compiled Java classes and Excel templates are excluded from the final GitHub tree.
- Production method IDs, versions, parameters and completion records are preserved.
- The immutable active production release is not overwritten by development-document edits.
- A source upload does not automatically launch, resume or advance any batch.
- No new native Windows run or Mac/Windows parity pass is claimed.
- The historical held 47-sample queue is outside this work and is not resumed or rewritten.
- Existing local source is not replaced with a new local copy of the external-drive package.

### Publication-only verification and packaging

- Parsed249 archived Python files without syntax errors.
- Emptied three styled Excel templates across eight sheets and passed exporter round-trips during packaging; those workbook copies are retained outside Git after the user chose text-only publication.
- Preserved the live templates and all original scientific result artifacts.
- Made the publication fixture path relative and included CHANGELOG.md in the publication freezer.
- The fresh fixture-suite attempt was incomplete because of unavailable/stalled local dependencies; some synthetic fixtures passed, but no new54-test pass is claimed.
- An unpushed deployment candidate passed its dependency-free release-manifest check. The final text-only archive excludes that bundle/pointer and is not native Windows or biological-data validation.

## 2026-10-30 — Shared modular pipeline v1

### Structure and migration

- Introduced one common implementation for TRAK1, TRAK2 and future proteins.
- Kept dataset identity, channels, acquisition units and methods in configuration.
- Separated thin platform launchers from shared scientific implementations.
- Defined StageContext inputs, stage-owned output directories and StageResult outputs.
- Separated detection, refinement/SE, mask, assignment, distances, decomposition and bootstrap.
- Separated scientific statistics from presentation and publication.
- Made original-C1 arithmetic averaging an independent scientific product.
- Added explicit release selection and small frozen-code-manifest checks.
- Preserved legacy releases, existing results and launcher backups for rollback.
- Read legacy results without migrating or duplicating localization masters.
- Adopted metadata for 211 existing TRAK cells: 105 TRAK1 and 106 TRAK2.
- Metadata adoption did not mean that 211 cells were newly analyzed.
- Reused the four existing environments and configured Fiji/Java installations.

### Scientific cache boundaries

- Scientific identity uses explicit method ID/version, canonical parameters and upstream artifact IDs.
- Source/release checksums audit integrity; they do not define scientific cache identity.
- Scientific algorithms require method-version changes at their responsible stage.
- Full-image Nellie depends on original C1 and segmentation settings, independently of ROI.
- Support counts depend on broad masks, window/stride, actual participants and counting rule.
- Thresholding depends on saved counts and the chosen support threshold.
- Changing only the threshold can reuse compatible counts.
- Changing the window, frame participation or counting algorithm requires new counts.
- Foreground cleanup and enclosed-hole filling are independent of support counting.
- ROI intersection is separate from full-field segmentation and cleanup.
- ROI changes preserve full-field Nellie but affect ROI-dependent fitting and selection.
- Reassignment preserves compatible detections, fits and masks.
- Label changes invalidate affected group distance counts.
- Coordinates, valid UIDs, endpoint frames, tau and complete bin edges bind distance identity.
- Decomposition changes can reuse compatible raw distance counts.
- Resampling changes regenerate the corresponding bootstrap products.
- Colors, fonts, layout and column order are presentation parameters.
- Filtering, smoothing, binning, normalization, membership and averaging remain scientific operations.

### Mask, UID and unknown contracts

- Masks declare TYX shape, original frame IDs, time mapping and participating frames.
- They declare pixel centers, pixel/time units, crop offset and C1/C2 registration.
- Native coordinates use integer pixel centers, closed unit squares and zero crop offset.
- Identity registration remains explicit and independently unverified.
- Missing frames are not replaced by zero masks or interpolated observations.
- Incomplete support blocks make affected assignments unknown.
- A complete empty mask remains a valid empty observation.
- Original same-frame C1 zero pixels and FOV boundaries retain unknown precedence.
- Contradictory coordinate/time metadata stop the affected task rather than guessing a transform.
- Detection source identity and native detection ID define persistent UID.
- Sorting, current row number, refined coordinates and labels cannot define UID.
- Legacy reads check UID uniqueness, correspondence, frame/time units and SE semantics.
- The fit master is read-only; new methods write UID-keyed label tables.
- Missing information or ambiguous error semantics are reported rather than inferred.

### Distances, bootstrap and command behavior

- The v1 distance method supports tau=1 original frame only.
- Other tau values require a separately versioned implementation, not a configuration-only change.
- Retained counts use original-frame anchors, endpoint validity and zero-removal records.
- Pooled full-film histograms alone are not a sufficient temporal-bootstrap cache.
- Resampling records retain source/target frames, blocks, strata, shared weights and replicate IDs.
- No adjacent pair is fabricated across sampled block joins.
- Plan reports reuse, missing inputs and recomputation without executing science.
- Run explicitly performs requested missing science, summaries and publication.
- Summarize --completed-only computes hierarchy statistics from completed cell products.
- It never starts missing localization, segmentation or single-cell TARDIS.
- Refresh --completed-only only publishes saved scientific products.
- Missing scientific products are reported; refresh does not fall through to run or summarize.
- Both successful and missing-input refresh require zero scientific stage calls.
- Scientific completion and publication completion are recorded independently.
- Missing PNGs/workbooks do not discard completed fits, masks or TARDIS.
- New modular stop requests do not resume or alter older queues.

### Recorded acceptance

- The saved modular-v1 release notes report 54 bounded fixtures passing.
- T5 passed the fixed ten-frame identity and geometry audit.
- Its 22 saved group/scope curves and three bootstrap plans reproduced from retained counts.
- A nested-process interruption fixture left no running child.
- Windows launcher quoting was checked statically.
- These are historical acceptance records, not a new November 2 replay.

## 2026-10-30 — Publication recovery and hierarchy reporting

- Verify generated artifacts with byte size and recorded content fingerprints.
- Retain modification time as provenance rather than scientific-content identity.
- Close writable mask mappings before capturing committed completion metadata.
- Recover 105 saved TRAK1 publications without rerunning scientific stages.
- Restore the missing root Intra navigation and normal refresh publication chain.
- Record 13 TRAK1 Intra galleries with 78 canonical PNGs and 78 On/Not-on overlays.
- Record 118 On-percentage/ROI-area PNGs and 118 reader-facing Excel workbooks.
- Check counts, area definitions, normalization order, weighting, SD/SEM and workbook values.
- Preserve full precision in machine-readable tables.
- Keep run IDs, status/path fields and calibration provenance outside reader tables.
- Per-cell selected total means On + Not-on + unknown, not every detection candidate.
- On% remains 100 * On / (On + Not-on); unknown and QC excluded are separate.
- BR reports contain participating cell values and equal-cell means.
- Condition count reports retain pooled per-cell and equal-replica means.
- Pooled per-cell count means give every cell one weight across the condition.
- Equal-replica count means average the BR cell means with equal BR weight.
- Reader BR/condition count columns report means per cell, not sums over cells.
- On-percentage and area condition summaries retain equal-BR weighting.
- First ROI area refers to the frame 0–49 support mask, not instantaneous frame-0 segmentation.
- Full-film mean ROI area is time weighted over all required mask blocks.
- Missing required blocks make the full-film mean unavailable rather than shortening its denominator.

### Historical Intra aggregation — v1 area rules superseded on 2026-11-02

The following bullets record v1. Its area-view order is superseded by summary
method v2 above; the signed and peak rules remain unchanged.

- BR canonical signed Intra is the arithmetic mean of eligible cell signed curves.
- Localization counts and pair counts do not weight the cells.
- BR area normalization divides the canonical signed mean by its signed area.
- It does not substitute a mean of individually area-normalized cell curves.
- The BR internal peak view averages individually positive-peak-normalized cell curves.
- Condition signed curves average canonical BR curves with equal BR weight.
- Condition area/peak views normalize each canonical BR curve before averaging.
- The final condition mean is not forced to peak at one again.
- On, Not-on and all three views share an eligible membership roster.
- Invalid/missing curves are excluded with reasons, never filled with zero.
- Condition output retains BR mean, sample SD, SEM and participating counts.
- One valid BR has no between-BR SD/SEM band.
- Full-range and zoom comparison plots retain compatible units and shared y limits.
- BR, condition and top levels include On/Not-on same-axis overlays.
- Temporal-bootstrap intervals are not mixed into between-BR variation.
- BR names do not automatically pair proteins or conditions.
- Core hierarchy summaries do not add formal significance tests.

## 2026-10-26 — Independent TRAK1 Windows package and reliability fixes

- Created a separate 105-cell TRAK1 inventory: WT 34, TDK 37 and MDK 34.
- Preserved original identifiers, including T21 and T61.
- Kept TRAK1 configuration, state, caches and completion registration separate from TRAK2.
- Reused installed sibling TRAK2 Windows software read-only.
- Added a TRAK1-only runtime repair route for changed drive/home paths.
- Preserved START, RESUME, Control-C and stop-after-current-cell entry points.
- Removed native Mac/Windows parity and pilot prerequisites from direct batch start.
- Retained ten sample timepoints and necessary size, ROI, QC, support and count checks.
- Wired Intra, overlays, scalar metrics and Excel into one completion/reuse chain.
- Added publication-stage failures, missing-output lists and publication-only retries.
- Checked 38 expected default nonempty PNGs separately from scientific completion.
- Fixed concurrent JSON writers with unique same-directory temporary files.
- Parallel stages use separate Windows-safe active-phase records.
- Corrupt optional status no longer hides the original failure traceback.
- Normalize equivalent DOS and extended paths after physical resolution.
- Preserve component-based containment and rejection of writes outside the portable data root.
- Include resolved-root/target diagnostics in future path failures.
- Saved history reports 26 Windows path cases plus concurrency regressions passing.
- These orchestration fixes did not change scientific estimators or QC.
- The package-delivery operation started no scientific TRAK1 cell on Mac.

## 2026-10-23 — Windows migration, axis histograms and direct start

- Prepared a separate Windows x64 release while preserving the Mac launch method.
- Set bundled Java paths before ScyJava initialization in controller and worker.
- Added separate X/Y nominal-SE histograms for All, On and Not-on.
- Retained 0–200 nm, 5 nm bins, linear counts and the original 2D precision plot.
- Backfilled X/Y PNGs for ten completed WT TRAK2 BR1 cells from saved counts.
- Added cache-based BR/condition Intra summaries to both platform packages.
- Marked progress explicitly when only a partial roster had completed.
- Restored T4's original figure-directory name at the user's explicit request.
- Preserved qualification failures rather than fabricating a native pass.
- Made cross-platform qualification optional for direct START/RESUME at the user's request.
- These changes preserved the Mac scientific method and original results.

## Isolated WT–TRAK2 / BR1 / T5 experiments — 2026-10-26

- Preserved default masks, fits, QC, ROI and original 1-SE labels as the baseline.
- Tested 1/2-pixel distance-to-mask assignment on all 2500 frames.
- Use floating centers, closed foreground pixel squares and tangency inclusion.
- One/two pixels equal 117/234 nm; these are spatial tolerances, not SE estimates.
- Original On/Not-on: 32,489 / 59,371; On%=35.367951%.
- One-pixel On/Not-on: 39,230 / 52,630; On%=42.706292%.
- Two-pixel On/Not-on: 45,931 / 45,929; On%=50.001089%.
- Selected=91,860, QC excluded=2,344 and unknown=0 remained fixed.
- Original 1-SE On need not be a subset of 1-pixel On when some ellipses are larger.
- Recomputed TARDIS for both saved radius labels using the original numerical rules.
- Saved full/ten-segment results with 1000 shared L10/L25/L50 draws.
- All-selected Total/background counts matched the saved baseline.
- Separately tested fixed C1 contrast on blocks 0–49, 1200–1249 and 2450–2499.
- The input mapping was round(65535 * clip((C1 - 402) / 338, 0, 1)), saved as uint16.
- The 150-frame contrast pilot completed with seven PNGs and a 150-page viewing TIFF.
- A later full-film contrast directory retains numerical and plot completion records.
- Its retained plot marker says visual review pending; this is not promoted as a validated default.
- November 2 publication does not resume that experiment or rewrite its stored status.
- No experimental radius or contrast method became a cohort-wide batch default.

## Isolated MDK–TRAK1 / BR1 / T1 experiments — 2026-10-30

- Only this cell was tested; this is not full-MDK, WT or TDK experimental validation.
- Pilots used complete blocks 0–49, 1200–1249 and 2450–2499.
- Compared per-frame 1–99% contrast stretching with a lower-cutoff-only variant.
- Upper clipping reduced coverage inside some bright plateaus under the tested Nellie settings.
- Lower-only clipping and larger scales changed boundaries as well as interiors.
- Additive-scale tests preserved original scales and appended scales to nominal 1.5/2.0 µm.
- These visual changes did not establish a segmentation-accuracy ranking; filter radius is not a hard organelle-size cutoff.
- The strong-signal experiment used original C1 with no intensity rewriting.
- For each full frame, compute median m and MAD from positive raw pixels.
- Define strong pixels by C1 > m + 6 * MAD / 0.6744897501960817.
- Union these with per-frame Nellie broad before unchanged support and cleanup.
- This robust intensity spread is not calibrated noise or a 6-sigma confidence statement.
- Completed 2500 frames, 50 support masks and original 1-SE reclassification.
- Original On/Not-on: 11,482 / 20,804; experimental: 11,659 / 20,627.
- On%: 35.563402% → 36.111627%, a +0.548225 percentage-point change.
- Exactly 177 Not-on observations became On; original On was retained.
- Selected=32,286, QC excluded=728 and unknown=0 were unchanged.
- Mean ROI area: 205.440953 → 206.962896 µm², approximately +0.74%.
- Recomputed full/ten-segment TARDIS and 1000 shared L10/L25/L50 bootstrap draws.
- All-selected pair counts stayed identical; only group partitioning changed.
- Saved signed, signed-area and full-range positive-peak normalized comparisons.
- Normalized curves largely overlapped; signed On amplitude changed modestly.
- Saved 16 PNGs; bounded audit covered ten raw frames and retained numerical products.
- Retained negative bins, unavailable curves, original UIDs, coordinates, SE and QC.
- Publisher scientific calls were zero; other cells and production defaults were untouched.

## Production scientific settings preserved

This table records current TRAK1/TRAK2 configuration, not a new calibration.

| Component | Retained setting |
| --- | --- |
| Acquisition | 2500 frames; TYX; 684×428; 117 nm/pixel; 20 ms/frame |
| Registration | Identity; independently unverified |
| Detection | TrackMate 7.14.0 LoG; radius 0.28 µm; threshold 6.5; median/subpixel enabled |
| Refinement | 7×7 integrated Gaussian with external annular-background constraint |
| Annulus/support | Inner/outer radii 6/12 px; neighbor core 4 px; ≥80 retained pixels; ≥4 sectors |
| Fit settings | Width 0.5–3.5 px; starting sigma 1/2 px; ≤350 iterations; original QC |
| Nominal camera | Offset 100 ADU; 0.46 e/ADU; EM gain 1; read noise 1.6 e RMS |
| Error | Full observed-Hessian nominal per-axis SE including nuisance parameters |
| Nellie | Recorded v1.0.4; original full C1; filter radii 0.25–1.0 µm |
| Native labeling | Min radius 0.117 µm; native min component 4 pixels; no additional dilation |
| Support | Nonoverlapping 50 frames; sum Boolean pixels; threshold ≥10 |
| Cleanup | FG8/min32; enclosed BG4 holes <64; then separate original ROI intersection |
| Assignment | Original diagonal k=1 SE ellipse; closed pixels; tolerance 1e-12; unknown priority |
| TARDIS | Original tau1; 0–3 µm; 300 bins; both endpoints in the same group |
| Zero distances | Cross-frame zeros retained; all exact same-frame zeros removed |
| Normalization | Independently normalize pooled Total/background on 0–3 µm |
| Beta | Tail least squares 2.1–3.0 µm; clipped [0,1]; raw beta retained |
| Tail support | ≥100 Total and ≥100 ordered background tail pairs |
| Signed shape | Retain negative bins; abs(area)≤0.01 unavailable |
| Bootstrap | 1000; seed 20261007; 250-frame strata; L25 main, L10/L50 controls |
| Bootstrap pairing | Shared IDs/weights; source-block-internal pairs only |
| Intervals | Exploratory pointwise 2.5–97.5%; joint-valid fraction threshold 0.95 |
| C1 mean | Original arithmetic mean including zeros; independent cache |
| Histograms | Linear signal axis; separate X/Y and 2D SE; no precision-size filter |

## Runtime and interpretation limits

- Nominal camera parameters and unverified registration remain explicitly labeled.
- Higher On%, larger masks or similar JDD/Intra curves do not prove better segmentation.
- Temporal intervals condition on fixed masks/labels; segmentation/registration uncertainty is not propagated.
- Infrastructure changes do not establish binding, genotype effects or formal significance.
- Original data, result products and detailed machine provenance remain on the external drive.
- The full MDK comparison reused frozen numerical function bodies.
- Its successful entry uses CSV I/O after local pandas/library file reads stalled.
- Distance calculations use the same SciPy native Euclidean kernel, loaded directly.
- CSV wrappers change I/O only; unavailable/blank values are never replaced with zero.
- Rendering used an existing system Python; no dependency was installed or modified.
- Stored TIFF byte order was converted losslessly for display and matched original sample fingerprints.
- Experimental wrappers are not newly qualified native Windows paths.
- Third-party sources and dependency versions remain attributable to their original packages.
- Entire installed Python/Fiji/Java environments are not redistributed in the source snapshot.
- Raw data and result exports remain excluded from GitHub publication.
- Historical fixture/audit/completion records remain on My Book as evidence.
- Documentation changes alone do not change method versions or invalidate compatible scientific caches.


Completed on 2026-11-02: six focused regression tests passed; corrected saved-curve hierarchies were regenerated for all 105 TRAK1 and 106 TRAK2 cells (26 nodes, 312 PNGs). Prior cohorts and every signed/peak saved row were unchanged. Upstream cell-science calls, new cells and scalar/Excel rewrites were zero. V1 numeric result directories are retained.


BR4/BR5 preparation checks completed on2026-11-02:37focused tests passed (7inventory,4legacy-source,14summary/normalization,12batch-control). Two real committed legacy-cell adapters read successfully under the isolated new configs. These were code/metadata checks only:0newcells and0scientific stages started. The new queue is pinned to its own frozen release20261102T_BR4_BR5_batch_v1; the prior global release pointer is retained.
