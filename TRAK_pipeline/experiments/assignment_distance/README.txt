Assignment-distance sensitivity pilot

Standalone entry: compare_assignment_distance.py --cell <cell directory> --radii-px 1 2
Use the existing cleanup Python listed in the active release runtime_paths.json. This pilot accepts WT_TRAK2_BR1_T5 only and never starts other cells or TARDIS.
Optional --run-id reuses a matching comparison run. Completed classifications, plots and workbook are reused; inputs and original outputs remain unchanged. Checkpoint row shards are discarded only after final independent validation; checkpoint records are retained.
Source selection requires a unique completed batch result, or an explicit --source-output. Output figures are in Analysis_Figures_distance_sensitivity_<runID>; numerical data, reader workbook and provenance are in analysis_results/distance_sensitivity_<runID>.
The two radii are geometric proximity tolerances of 117/234 nm; original localization standard errors remain unchanged.

The standalone entry also generates fixed1px/2px Not-on overlays on the source full-movie C1 mean cache. It reuses the cached arithmetic mean; no raw-movie averaging or classification is repeated for these viewing images.
