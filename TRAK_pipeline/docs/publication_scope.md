# Source publication scope — 2026-11-02

## Included

Shared source and compatibility kernels; relative dataset configs; Mac/Windows launcher generator and examples; four existing Windows dependency definitions; bounded fixture tests; architecture and change history; opt-in experiment source. This GitHub upload contains text source and documentation only. Java source is included; compiled Java classes, bootstrap NPZ assets and Excel templates are excluded at the user's explicit request. The existing My Book production copies retain these resources.

No active-release pointer or frozen deployment bundle is uploaded. A GitHub clone opens development source until an operator prepares resources and creates a new frozen release. Existing production release selection on My Book is unchanged. Source fingerprints audit publication and never determine scientific cache identity.

## Retained on the data drive

Raw TIFF/ROI/MAT/localization inputs; scientific masks/counts; fitted localization tables; saved cell/group labels and curves; bootstrap results; figures; inventories, completion registries, runtime logs/caches/checkpoints; installed Python/Fiji/Java and historical production releases. Compact verified case-study numbers appear in the change log, but the underlying observations and result workbooks are excluded.

## Deployment

The repository also retains its previous SRM simulation and MAT/locb TARDIS workflows. The real-image package is under TRAK_pipeline/. For the existing drive, place it as _TRAK_pipeline beside Xinran_TRAK1 and Xinran_FullsetTRAK2, retain existing dataset control directories, and configure existing runtime profiles. Cloning alone does not provide scientific data, inventories, installed software or host paths. Supplied relative configs reference the preserved legacy runtime profiles on that drive.

Prefer tools/install_launchers.py --dry-run before installation. launcher_examples/ holds examples of installed relative wrappers. No software installation, pilot or cross-platform numerical/pixel qualification occurs silently at normal startup.

Experimental configs use <CONFIGURE_EXTERNAL_PATH> placeholders. Configure them in a new isolated directory with the required baseline products. Some are Mac prototypes; unmodified Windows execution is not claimed. The successful full-movie experimental TARDIS entry is run_hybrid_tardis_csv.py; the original pandas-import entry remains audit-only.

## Publication-only adjustments

The final upload was narrowed to text source and documentation after explicit
user confirmation. The previously prepared blank workbook copies, Java binaries
and bootstrap reference NPZ stay outside the Git tree. This source archive is
not a self-contained deployment package: use the retained drive resources or
rebuild/provide them before freezing and installing an operational copy.


- Locate the analysis-report fixture package relative to its source, rather than the author's mounted volume.
- Remove an absolute historical parameter-reference path from published native_parameters.json; numerical values are unchanged.
- Empty report templates of measurements, preserving headers/styles and generic titles. Live templates and saved reports are not replaced.
- Retain the source freeze tool, but exclude publication-only deployment bundles and their pointer. The drive's active production release and existing scientific artifacts remain unchanged.

No production batch is started by publishing this archive. Packaging/Mac fixture checks do not claim native Windows execution of archived experiments.

## November 2 publication checks

All249 archived Python source files parsed without syntax errors. Three cleaned
Excel templates across eight sheets passed ZIP-content, visual and existing
exporter round-trip checks. A prepared, unpushed deployment candidate passed the dependency-free manifest
checker, but that binary-resource candidate is excluded from the final text-only
upload.

A fresh full fixture-suite rerun was not completed in this publication runtime:
the system interpreter lacked tifffile; a retry using already installed optional
packages stalled during dependency loading/rendering and was stopped before completing the suite. No
packages were installed and no production cell computation was started. Some
synthetic fixtures completed successfully; their temporary products are excluded
from the upload. The54-fixture result
in the historical change log is the recorded October30 result, not a new
November2 pass. Publication artifacts and attempt logs remain on My Book.
