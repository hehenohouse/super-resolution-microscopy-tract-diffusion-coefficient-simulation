# Change log

This log separates production changes, reporting repairs and isolated experiments.
Historical dates describe recorded work, not new analysis during publication.
The November 2 update prepares English source documentation for GitHub; scientific data remain on My Book.

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
