# Common protein analysis pipeline

One shared, frozen implementation serves TRAK1, TRAK2, Mac and Windows. Dataset identity, channels, acquisition units and scientific settings live in `configs/`. Existing scientific results and legacy releases remain in their original dataset folders.

This package does not start a batch when installed. It uses the four existing Python environments and the existing Fiji/Java installation; it installs no dependencies and performs no Mac/Windows numerical or pixel replay.

## Documentation and source publication — 2026-11-02

**GitHub scope: text source and documentation only.** Java.class, NPZ resources,
Excel templates and deployment release bundles are not included. The live My Book
package retains them. The source archive alone is not a turnkey installation;
provide/rebuild runtime resources before freezing and installing a new copy.



The current source archive includes the shared production pipeline, both-platform launcher examples, existing dependency definitions and opt-in experimental entry points. The detailed [change log](CHANGELOG.md) records both changes and retained behavior; [publication scope](docs/publication_scope.md) explains the code-only archive.

The datasets contain 105 TRAK1 and 106 TRAK2 cells. The modular migration adopted their saved results. This documentation/publication update does not re-run those cells. Run, summarize and refresh remain separate operations.

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
