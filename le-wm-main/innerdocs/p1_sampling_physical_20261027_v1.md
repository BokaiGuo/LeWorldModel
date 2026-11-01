# Frozen sampling winners → physical calibration consumer

NEW producer only. No old source/results/docs modified. Review/implementation
thread executes saved-array checks, mock tests and metadata prepare only; MAIN
alone may authorize real physics. No model/encoder/GPU/CPU neural work exists
in this runner. experiment-code skill informed fixed runs/logging/tests; its
external reference files are absent, and training/plotting is out of scope.

## Protocol

Two original calibration2 source slots × fixed `cem,iid_random` = four branches.
No zero/witness relabeling or physical reselection. Each selected original ten
raw commands executes at most once. Complete census: **8 resets, 40 public
steps, 44 explicit renders, 12 saved PNGs** at branch0/5/10. Each branch uses
fresh initial RGB, then fresh RGB after EVERY returned command before the next
dispatch. Renders between checkpoints are guarded/hash-journaled, not stored as
additional PNGs. This is the unchanged sealed RgbObservabilityGuard.

The implementation explicitly reuses sealed `p1_witness_physical.branch`,
`Store`, `raw_commands` and diagnostics by import, NOT global monkey-patching.
It does not call the old witness collect/selection/run or read witness outcomes.
New NAME/schema, contract, four-arm run census, prepare and complete source
manifest are independent. Old branch emits per-selection events without old
top-level schema. The final report has new schema and sampling-specific costs.

Actual sampling report external SHA and original prepared external SHA bind
all four arm report/journal files. Complete single gzip streams are boundedly
decoded; all60 full-context candidates, summaries, stable minimum, exact
independent nominal/whole/segmented replay, same task/seed and complete operation
receipts are validated. Raw commands must already be exactly float32 bounded
10×2; no clipping/conversion repair. Plan retains prediction, prediction SHA,
original winner SHA, source metadata and goal/Q/full-context binding.

Reset recipe: selected original random seed → regenerated native goal →
original sample_state using current block shapes and original slot/proposal/
attempt → original placement seed. Saved poststate is ONLY equality evidence,
never reset input. Before command1 check exact state/goal/RGB/goalRGB/root and
state-at-render. Source closure/model/task/raw-source hashes are all bound.
Plans are in original source slot order, NOT gain order. Root/method census is
exactly two×two; future results never change the plans.

Native terminal/truncated always stops. Return failures, render/guard failures,
wrong initial identity and early termination retain partial evidence, no retry,
replacement or pretend full endpoint. Native angular formula remains unchanged;
wrapped diagnostic and negative-angle/disagreement flags remain separate. A
native/wrapped disagreement is a protocol risk, not permission to continue.
Step error remains first error when env.close subsequently fails.

Eight completed resets imply16 reset-internal physics calls and8 goal renders
under bound original reset semantics. These are explicitly **implied**, not
instrumented counts. Public-step physics substeps are unknown here; 40 public
steps are not40 internal substeps. No extra physics is executed to measure them.

All output shares8MiB cap + per-write200MiB free floor. Preflight reserves full
8MiB plus200MiB. Exclusive new paths, fsync per journal/PNG/report write, hash
chain and final journal hash. Media/full-disk failure can prevent final report;
no quota relaxation/deletion or silent candidate reduction. Prepare stores only
small hash manifests/selected evidence, no model/large source closure copies.

## Evidence boundary

Only calibration comparison. Same two reused roots are not new independent
samples; padding history is not observed transitions. P1_admission and
model_task_eligible false, independent_confirmation false, neural_calls0.
Model-cost winner is not physical success; no threshold authority, plateau,
mechanism or efficacy admission is claimed. Sampling compute is inherited,
not newly free; it matches witness forward-call budget only, not total compute.

## Commands (project root le-wm-main)

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_sampling_physical_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_sampling_physical_20261027_v1 prepare --sampling-prepared results/p1_matched_sampling_cuda_20261027_v1_prepared01 --sampling-prepare-sha 1c4ee801b9e67b2529e23c51fae94ce0c7dd421290c8fd3edc2de6786f2ab0b3 --sampling-report results/p1_matched_sampling_cuda_20261027_v1_cuda_main01/report.json --sampling-report-sha cfd13ca478603c90c56b81c70f0ed161ef56d6a8de87da4c0e317388bd138e4e --output results/p1_sampling_physical_20261027_v1_prepared01
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_sampling_physical_20261027_v1 verify --prepared results/p1_sampling_physical_20261027_v1_prepared01 --prepare-sha PREPARE_SHA
# MAIN ONLY; recheck disk gate, use actual PREPARE_SHA printed above:
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_sampling_physical_20261027_v1 run --prepared results/p1_sampling_physical_20261027_v1_prepared01 --prepare-sha PREPARE_SHA --output results/p1_sampling_physical_20261027_v1_main01 --authorize-physics
```

Mock fixtures reuse the old test's array-only fake, not gym/Pymunk. Separate
saved-evidence tests read actual sampling files without neural/physics work.
Actual physics is not performed by this implementation thread.
