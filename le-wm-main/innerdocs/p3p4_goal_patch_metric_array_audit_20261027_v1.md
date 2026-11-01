# Independent saved goal-patch metric audit

Pure NumPy arithmetic plus original-goal PNG inspection. No imports of the metric
compiler, old planner numerical helpers, native_cost, neural model, or simulator.
Only the existing source-seal I/O helper is reused. This does not overlap fresh
double replay or a physical bridge. Existing sealed sources/reports are read-only.

## Frozen arithmetic contract

Saved native goal_vectors are ordered original/clean/four fixed corner sprites/
row-major object-cell deletions. legacy_goal_vectors must match the independently
bound original compilation report; its goal0 must equal the original packet goal.
Reconstruct old Q from the four legacy differences using its original
max(shape)*eps*largest-singular-value cutoff. Reconstruct the nine-column nuisance
span from four legacy differences, original-clean, and four corner-clean vectors.
Project each object-cell-minus-clean vector off that span, then SVD the residual.
Both new-span cutoffs are max(1e-6*largest singular, max(shape)*eps*source Frobenius
norm); the object floor uses the UNPROJECTED object directions. Zero task rank
refuses. No fallback, rank tuning, outcome selection or target replacement.

Compare reconstructed projectors and singular values at atol1e-10/rtol1e-8.
These tolerances only assess saved-array numerical consistency. Basis column signs
are not identities; projectors are compared instead. Save the reconstructed old,
nuisance and object bases and both Q matrices as SHA-bound float64 NPY files.
Three saved-array SVDs per root are newly performed (12 total for four roots),
distinct from inherited 8 metric SVDs and the upstream exporter validation SVDs.

For every source union candidate, root/input hash/H/index/full 5H-by-2 actions and
full terminal context must match the bound original uniform/dynamic bank. Duplicate
keys within an arm, conflicting shared witnesses, missing/extra candidates, partial
branches, H outside 2/3, or shortened actions refuse. Recompute BOTH costs as
sum((Q@(terminal-goal0))**2)/192 and independently resolve stable cost/H/index
argmin. Winner identity alone is insufficient: the full selected row must match.
Retain all candidates, raw actions, context/action hashes, both costs and both
winners. New selection is never executed; no commands are substituted or exported.
Different Q scales must NOT be compared as effect sizes.

## Goal-only mask provenance

The mask reconstruction function accepts only one independently SHA-bound original
224x224 uint8 RGB goal. It uses exact canonical object color [119,136,153], includes
every intersecting 14x14 cell row-major, and checks pixel counts, mask/RGB hashes,
sprite bounds, white perimeter, lack of object overlap and fixed-corner eligibility.
The complete catalog must match both frozen preparation and actual report. No
current, predicted, endpoint, physical outcome or pose image is a mask input.
This checks the frozen RGB intervention specification and saved encoding order;
it cannot independently regenerate neural embeddings or prove which pixels an
unobserved encoder execution consumed. No global invariance/pose semantics claim.

## Integrity, costs, and commands

Preparation verifies source compiler/planner seals and original bound data, legacy
vectors, original goal PNGs, journal, root shards, protocol IDs and report SHA.
It freezes this audit's new module/test/doc plus the source-seal helper. Audit
rechecks all bindings before/after arithmetic; root completions are fsynced and
failed computations retain an exclusive failure report without modifying inputs.
Original compile journal encoder/projector entries and declared budgets are
reconciled, not counted as new neural work: 80 encoder plus 80 projector forwards
are inherited; candidate generation cost is upstream/inherited, not free.
No model/physics calls, training, parameter fitting, downloaded data or deletion.

Run from le-wm-main:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3p4_goal_patch_metric_array_audit_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p3p4_goal_patch_metric_array_audit_20261027_v1 prepare --compile-dir results/p3p4_goal_patch_metric_20261027_v1_cuda_main01 --report-sha256 7b38b91e18497b625f36e7c52d59c07e53204e6047af083c10b4c4631aa2fee6 --output results/p3p4_goal_patch_metric_array_audit_20261027_v1_prepare_v1
```

Retain the resulting prepare.json FILE SHA independently, then:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3p4_goal_patch_metric_array_audit_20261027_v1 audit --prepared results/p3p4_goal_patch_metric_array_audit_20261027_v1_prepare_v1 --prepared-sha256 RETAINED_SHA --output results/p3p4_goal_patch_metric_array_audit_20261027_v1_main_v1
```

Only new exclusive output directories are accepted. This is known-root calibration,
not P1/P3/P4 admission, no force/physical causal conclusion and no task-benefit claim.
The downstream gap remains independent selected-path replay and a separately
reviewed physical consumer, owned by another thread, not supplied by this audit.
