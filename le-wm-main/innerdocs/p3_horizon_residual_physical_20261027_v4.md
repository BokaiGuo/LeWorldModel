# Independent mixed-H2/H3 physical consumer v4

Implementation-thread predeclared under authorized implementation. Main owns
actual physics and HOST CUDA. No training, planner edits, replacement roots,
new search, outcome-selected budget/threshold or admission. P1/P3 remain false.

## Source and report binding

Only consumes the original sealed v4 planner exporter. Its source SHA is
27f8b7f1cabed21d664ded40be0e0640aed9b0367bdfe2c39d377f4c482463c0;
its prepare receipt SHA is
1374e31190e18fabadc90f8fe8668e3800c863be5083ced8a4fa9bf023a85b98.
The archived source must already contain selected_actions; no retrospective
exporter patch or old preparation reseal. Old consumer runners are not called.
Only durable low-level branch execution, source sealing and source references
are reused. Prepare and bind import neither simulator nor neural runtime.

Two immutable stages: --prepare freezes consumer/test/doc, transitive source,
planner preparation and four source context/goal0/Q/RGB references before a
completed planner report is needed. --bind consumes an explicit actual report
byte SHA and calls the frozen exporter to create eight plans in a NEW path.
Source preparation is never rewritten. --verify on bound preparation repeats
all source checks and exporter validation. Physics refuses source-only stage.
No placeholder candidate or missing-report success. Bind stores planner report,
execution protocol and all four actual neural journals as immutable inputs.

All eight duplicate-free root/mode identities must match the same four roots.
Each export must preserve original context, goal0, Q, source packet, root index,
all 10/15 commands and report/protocol/selection hashes. H2/H3 is the selected
path horizon, not maximum search horizon. Plan SHA binds all these fields.
Fixed root-index counterbalanced arm order is inherited, never outcome-sorted.
Upfront source RGB references are checked for all roots; runtime reset checks
history/goal RGB and paired identical full initial state/history/goal identities.
Every warmup raw history must be bounded finite (3,2). Its constant-five-step
packets use original public float32 mean/std; tokens[1:] must match the two
predictor-context incoming tokens EXACTLY, not approximately. Runtime public
normalization constants must equal the source-frozen values. All four physical
roots must have finite seven-state and goal vectors and exact fingerprints before
any selected-path branch dispatch, not just before each individual branch.
No older physical endpoints are read or reused.

## Durable guarded physics

Exactly one selected open-loop path per root-arm. Same canonical RoyalBlue RGB
presence guard before every raw command, including H3. Simulator state is used
only for labels, never dispatch decisions. Reset/warmup cost is explicitly
separate from selected-path counters. Attempt is fsync-journaled before env.step;
returned step is charged before render/evaluation. Every RGB file is exclusive,
fsync'd and hash-bound. Every branch including reset, evaluator, render and
dispatch failures retains a durable physical receipt. Unattempted arms are
materialized as blocked, not dropped/replaced/retried. Journal/trace/command
identity is verified before proceeding. Report preserves all eight root-arms.

Full completion requires original durable helper status completed and all
selected commands returned with a fresh endpoint. Missing guard evidence,
environment termination, failed step or stale render cannot be promoted.
Freshness requires matching final RGB step and returned-step count, equal
attempt/return counts, and a known guarded/terminal/completed status. A valid
zero-step initial RGB after guard refusal is a partial endpoint, not a winner.
An exception after step but before render cannot encode the previous image.

Pairs are stratified as full_H2_pair, full_H3_pair, full_mixed_H2_H3,
partial_same_steps, or partial_or_unmatched. Initial context must match before
any pair is full. Full mixed duration is not same-horizon evidence. Pose delta
is reported only for full same-horizon pairs; mixed and partial rows retain
individual official/pose metrics without a pooled efficacy statistic.

## Optional endpoint encoding

All eight branch receipts and physical_report.json must be durable BEFORE any
CUDA preflight/model loading/forward. No model allocation occurs in physics
unless --encode-terminals is explicitly supplied. At most one fresh endpoint
per arm, eight total, batch1 HOST CUDA only. No stale endpoint encoding, no CPU
fallback. Physical failures remain independent of encoding failures.
Setup/encode attempts/results are fsync-journaled; receipts record attempted
sample count including failed forwards. Predictor is forbidden and counted.
Frozen model/config bytes and final model state are checked. Original source
goal0/Q score is computed from encoded actual endpoint, preserving full_H2,
full_H3 or partial_H2/H3 scope; it is not a new goal or action ranking.

No new predict, action Jacobian, VJP or SVD is required by physical execution.
The immutable exporter performs eight saved-array SVDs per verification call:
binding performs one export plus final validation (16 array SVDs); bound verify
performs eight; execute pre/post validation totals16. These are CPU verification
overhead, not silently neural-free total compute. Planner's actual shared costs
remain charged once in its own source report; consumer never adds standalone
attributions. Encoding is up to eight additional visual forwards, not budget
matching or P3 evidence. No cross-horizon/partial aggregation or total FLOP claim.

## Commands and limitations

Use module experiments.spectral_reachability.p3_horizon_residual_physical_20261027_v4.
--prepare --output NEW_SOURCE_PREPARED; then --bind --prepared SOURCE_PREPARED
--prepare-sha256 SOURCE_RECEIPT_SHA --planning-output ACTUAL_CUDA_DIR
--report-sha256 ACTUAL_REPORT_SHA --output NEW_BOUND_PREPARED.
--verify --prepared BOUND_PREPARED --prepare-sha256 BOUND_RECEIPT_SHA.
Reviewed main only: --execute-physics --prepared BOUND_PREPARED
--prepare-sha256 BOUND_RECEIPT_SHA --output NEW_PHYSICAL_DIR [--encode-terminals].
All paths are exclusive. Never rerun into an old output. Tests use saved source
evidence and fake environments/encode callbacks only, not neural/real physics.
