# Project change log

## 2026-11-02 — Shared real-image pipeline source publication

The authoritative [detailed TRAK pipeline change log](TRAK_pipeline/CHANGELOG.md) separately records additions, fixes, retained scientific parameters, experimental scope, output contracts, validation and deployment limitations.

Added the common real-image package, both-platform launcher examples, environment definitions, bounded fixture tests and isolated experiment source. Existing simulation and MAT/locb TARDIS implementation remains in its previous locations. No production analysis or default-mask change is started by publication; raw movies and individual-cell result tables are excluded. See [publication scope](TRAK_pipeline/docs/publication_scope.md).

Historical MAT/locb entries remain in [its change log](TARDIS/path_irrelevant_python/CHANGELOG.md).

The final publication scope is text source and documentation only, as explicitly
confirmed by the user. No compiled Java, NPZ or Excel binary assets are included;
the existing installed My Book package retains its resources.
