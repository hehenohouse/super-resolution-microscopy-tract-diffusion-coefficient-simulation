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
