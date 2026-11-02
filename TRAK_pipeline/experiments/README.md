# Opt-in experimental source archive

These entry points are separate from normal batch execution. They are not called by START_BATCH, summarize or refresh automatically. Only source and path-placeholder config examples are published. Data, caches and result artifacts stay on My Book.

| Directory | Scientific question and tested scope |
| --- | --- |
| assignment_distance/ | WT TRAK2 BR1 T5: original1SE versus1/2-pixel Euclidean proximity to unchanged full masks, plus separate complete TARDIS comparisons. Radii are spatial tolerances, not localization SE. |
| cluster_kymograph/ | WT TRAK2 BR1 T5: raw C2 X/time and Y/time profiles, fixed96-record cluster selection; no track linking or molecular-identity inference. |
| mask_sensitivity/c1_contrast_402_740/ | WT TRAK2 BR1 T5 isolated150-frame input-remapping test. Historical full-movie extension artifacts are not resumed by this archive. |
| mask_sensitivity/c1_imadjust_p01_p99/ | MDK TRAK1 BR1 T1,150frames: MATLAB-style lower/upper contrast clipping before Nellie. |
| mask_sensitivity/c1_imadjust_low01_nohighclip/ | Same150frame case, lower clipping without upper clipping. |
| mask_sensitivity/nellie_additive_scales/ | Keep all default scales, append larger scales, compare150frames. Installed Nellie remains a dependency; native-source reference snapshots stay on the drive. |
| mask_sensitivity/strong_signal_union_pilot/ | Per-frame original broad OR conservative raw strong pixels before unchanged support/cleanup,150frames. |
| mask_sensitivity/strong_signal_union_full_tardis/ | Same method, all2500frames of MDK TRAK1 BR1 T1; original1SE assignment and complete TARDIS/shared1000-repeat bootstrap. |

## Strong-signal recipe

For every original full raw C1 frame, compute m=median(positive pixels), MAD=median(abs(I-m)) over those same pixels, and s=MAD/0.6744897501960817. Strong pixels satisfy I>0 and I>m+6*s. Union them with the original per-frame Nellie broad mask before support>=10/50, FG8 min32, BG4 enclosed holes<64 and the original ROI. No intensity rewriting, dilation or blanket hole fill is added. The robust intensity spread is not calibrated camera/background noise or a6-sigma confidence threshold.

## Successful full-movie sequence

In strong_signal_union_full_tardis/source/:

1. run_hybrid_mask_assignment.py --config <configured-json>
2. prepare_hybrid_tardis_counts.py --run <isolated-run>
3. run_hybrid_tardis_csv.py --run <isolated-run>
4. render_full_comparison.py --config <configured-json>

The original run_hybrid_tardis.py is an import-only attempt retained for audit. The successful path uses stdlib CSV I/O around unchanged frozen numerical functions and the same native SciPy Euclidean kernel. Publication reads saved signed, area and peak-normalized scientific columns.

Examples require configured external paths and existing baseline products. Retain source/frozen-kernel directory structure in the isolated run. Never target an existing production output directory. Similar JDD/Intra curves do not establish segmentation accuracy. Experimental code is not a new batch default.
