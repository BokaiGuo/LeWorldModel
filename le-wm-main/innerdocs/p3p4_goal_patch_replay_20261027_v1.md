# Goal-patch selected-winner fresh replay v1

Main-thread predeclared under authorized implementation. This is an executable
fresh-witness exporter, not physical execution, metric efficacy, a P3 scheduler
repair, or P1/P3/P4 admission. All admissions remain false. No old physical
endpoint is loaded or re-labelled as a new outcome.

## Frozen objects and comparison

Consumes the sealed metric compiler's actual
`results/p3p4_goal_patch_metric_20261027_v1_cuda_main01/report.json`, its original
preparation, journals, goal vectors and per-root files. Reconstructs the old
planner's entire candidate union (not only selected arrays), duplicate shared
candidate equality, all costs, both metrics and stable winners. Root/context,
goal0, model, Qold/Qnew, candidate bank, raw commands and horizons are bound in
each exported plan. Four roots, eight arms, fixed counterbalanced arm order:
even roots old then new, odd roots new then old. No outcomes select roots,
thresholds, candidates or horizons. New winners H2/H3/H3/H2, old all H2; later
physical analysis MUST separate same-H complete pairs from mixed-H and partials.

## Fresh independent neural witness

Each arm starts twice from the original immutable full context: once with all
10 or 15 raw commands, once packet-by-packet. Entire latent_history (3x192) and
normalized action_history (2x10) must equal each other AND the saved candidate
exactly. Both old/new Q costs are rechecked. `selected_actions` exists before
seal and rejects incomplete execution, altered bank, source drift, missing arm,
wrong root/horizon/action, journal inconsistency or non-identical witnesses.
Downstream must use this NEW exporter, not an old consumer whitelist.

## Budget and failure semantics

Fresh replay: 36 predict +36 action-encoder entries; 108 context tokens for each
entry type. Per-path H, per-arm 2H, roots8/10/10/8 for each type. No padding or
refund. Predict includes native predictor and pred_proj, not separately hooked
or FLOP-measured. Visual encoder0, AD/Jacobian/VJP0, execution SVD0. Model weights
and eval/frozen CUDA state checked; no CPU neural fallback or downloads.

Each full model-free source reconstruction performs 16 saved-array SVDs:
8 original planner residual reconstructions +8 metric reconstructions. Prepare
and verify each reconstruct once; CUDA execution verifies before and after
(32 validation SVDs total); downstream selected_actions reconstructs once.
Reconstruction also rescales the whole bank with two Q costs per candidate and
two minimum selections per root; bank sizes and operation counts are bound in
the selection digest. These are CPU/source-validation overhead, not free work.
Inherited planner resource ledger and metric80 encoder +80 projector +8 SVD
are copied as shared costs; cached context acquisition remains unknown, not
zero. No total-resource or FLOP-equivalence claim.

Durable fsync journals record attempt before dispatch and completion/error
afterward. Per-arm receipts retain whole/segmented partials. Failure stops
subsequent dispatch, preserves eight-arm census with blocked rows, and cannot
be exported or automatically retried. Output directories are exclusive NEW.
All eight fresh physical branches must subsequently be executed; older physical
results cannot replace even unchanged old-Q controls.

## Main-only commands (repository root)

Use `.venv/bin/python -m experiments.spectral_reachability.p3p4_goal_patch_replay_20261027_v1`
with `--prepare --output NEW_PREPARED`, then
`--verify --prepared NEW_PREPARED --prepare-sha256 REVIEWED_SHA`, then only main
may run `--execute-cuda --prepared NEW_PREPARED --prepare-sha256 REVIEWED_SHA
--output NEW_CUDA_OUTPUT`. No simulator imports or visual encoding in replay.
Tests are saved-array/metadata fixtures, not neural or physical evidence.
