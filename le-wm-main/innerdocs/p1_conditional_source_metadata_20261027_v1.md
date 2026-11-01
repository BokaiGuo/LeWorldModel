# Historical source metadata resolver v1

Bounded streaming JSON projection hashes container bytes and decodes only the
explicit source identity/input-binding whitelist. Future outcome values are
syntactically skipped, not decoded, retained or used. The resolver does open report
containers; claiming it never reads their bytes would be incorrect.

Legacy census: top-level results/pusht* directories, report/protocol/root JSON and
source args. New census: top-level *request*.json/*plan*.json plus directories with
sourcepack.json/plan.json, irrespective of directory name. Own plan schema gates
slot extraction. Every declared parent seed and half-open reset span is reserved,
including rejected/unselected attempts. Sourcepack selected root hashes are also
excluded. Tests guard forbidden outcome-value decoding and non-pusht discovery.
An outstanding prepared request therefore reserves its pool; rerunning prepare
for the same pool fails conservatively. Execute its already prepared request,
not a freshly prepared duplicate. No code silently redraws the pool.

Bounds: 4096 directories/files, 64MiB per JSON, 256MiB total projection;
exceeding bounds errors instead of silently certifying coverage.
Empty/analysis-only directory names are not provenance failures.
Known used/reserved legacy seeds are excluded. Legacy missing actual reset traces
remain unknown; current-factory conservative envelopes are protection, NOT
historical authority. Sourcepack plan membership is conservative pool attribution,
not an exact per-root selected-seed association certificate.

This is NOT an independently reviewed globally complete authority. Omitted nested
locations and interrupted runs with journal-only records are not certified by this
census. Global confirmation stays unknown even after all known exclusions pass.
Calibration exploration is allowed with explicit scoped unknowns and known-reuse
rejection; it supplies no confirmation freshness or scientific admission.
