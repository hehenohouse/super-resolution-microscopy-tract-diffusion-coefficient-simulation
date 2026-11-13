# Windows Miro1 batch (new independent dataset)

## Start

With the entire My Book drive attached to Windows, double-click `START_MIRO1_WINDOWS.cmd` in the drive root. It uses the same Windows Python, Fiji, Java and dependency environments as the TRAK batches; nothing is installed.

The queue (`batches/20261113_Miro1`, entered through `batches/20261113_Miro1_all`; release v9) contains all 102 Miro1 cells in `Xinran_Miro1`:

| Condition | BR1 | BR2 | BR3 | Total |
| --- | ---: | ---: | ---: | ---: |
| WT | 10 | 11 | 12 | 33 |
| TDK | 12 | 12 | 12 | 36 |
| MDK | 10 | 11 | 12 | 33 |

The first cell must finish its science and all its figures before the rest of the queue continues. If it fails, the batch stops with its checkpoints kept; report the error before resuming. At the TRAK BR4/BR5 speed (median 20 minutes per cell) the whole queue takes about 34 hours.

Scientific methods are those of TRAK1/TRAK2 (TRAK1 BR4/BR5 configuration) except three mitochondrial-mask parameters chosen for Miro1 on 2026-11-13: Nellie Hessian gate threshold ÷ 4 (default 2), support ≥ 20 of 50 frames (default 10) and mask pieces ≥ 60 px (default 32). These are recorded in the batch manifest (`method_overrides`). Miro1 On rates are therefore not directly comparable with TRAK1/TRAK2 On rates computed with the default mask. Cells whose detection and fit are already committed (6 cells run on the Mac) reuse them; Nellie and later stages are recomputed. The earlier queue `batches/20261111_Miro1(_all)` (default mask) is superseded. Miro1 is summarized on its own at cell, BR, condition and dataset level; it is never pooled with TRAK1, TRAK2 or HaloTag-only. Miro1 BR1 shares its acquisition date with TRAK1 BR1; this does not make them paired experiments.

## Pause, resume and other actions

- `STOP_AFTER_CURRENT_CELL_MIRO1_WINDOWS.cmd`: finish the current cell, then stop.
- `RESUME_MIRO1_WINDOWS.cmd` (or Start again): continue; completed cells and committed checkpoints are reused.
- `PLAN_MIRO1_WINDOWS.cmd`: read-only count of queued, completed and pending cells.
- `UPDATE_MIRO1_SUMMARIES_WINDOWS.cmd`: rebuild the Miro1 summaries from completed cells (no cell science).
- `REFRESH_MIRO1_FIGURES_WINDOWS.cmd`: redraw summary figures from saved summary data.

Ctrl+C interrupts the active batch and its child processes; committed checkpoints remain. Do not unplug the drive until the console has exited. Cell-specific failures are recorded and the queue continues; infrastructure failures (drive disconnected, write failure, environment failure) stop the batch.

The console should show the same speed settings as the TRAK batches: `"workers": 10` (fit), `"chunk_workers": 3` (Nellie) and TrackMate threads 12.

## Output locations

As for TRAK1/TRAK2: cell figures in `Analysis_Figures_modular_<ID>/` beside each cell's C1, C2 and ROI; cell products and checkpoints in the cell's `analysis_results/modular_cache/`; summary figures and reader workbooks in `Analysis_Figures_Cell_Metrics/` and `Analysis_Figures_Intra/` at BR, condition and `Xinran_Miro1` level; top-level pages `INTRA_RESULTS.html`, `CELL_METRICS_RESULTS.html` and `CELL_METRICS_TABLES.html` in `Xinran_Miro1`. Queue state lives in `Xinran_Miro1/_batch_pipeline/batches/20261113_Miro1/` and `_TRAK_pipeline/batches/20261113_Miro1(_all)/`.
