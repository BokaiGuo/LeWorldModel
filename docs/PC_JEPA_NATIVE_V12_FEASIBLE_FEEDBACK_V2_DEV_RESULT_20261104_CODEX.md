# JEPA v12 constructed-feasible feedback: development result

## Execution and limits

On one newly constructed native PushT root, both v12 response and the
zero/carry CEM comparator completed three 15/10/5-step decisions, each
followed by five physical actions and fresh RGB plus legal proprioception.
All six decisions used 1,920 five-step prediction tokens. The saved v12
refinement had final batches of 9, 2 and 4 proposals respectively.

| Boundary | v12 agent distance | CEM agent distance | Object distance in both arms |
|---|---:|---:|---:|
| after warmup | 31.338 | 31.338 | 0 |
| after 5 actions | 32.486 | 59.231 | 0 |
| after 10 actions | 7.242 | 43.355 | 0 |
| after 15 actions | 1.633 | 0.287 | 0 |

Both satisfy the recorded joint-position and angle success rule at the final
boundary. This is a **single constructed goal**: the witness did not move the
object, and the two planners also ended with zero object difference. It tests
agent positioning and feedback plumbing, not contact reconfiguration, random
goal feasibility, a success-rate advantage or a PC gate. It does not establish
independent checkpoint or training-data provenance. On this root CEM's final
agent distance was lower, despite v12 reaching the numerical success rule
earlier; no performance claim follows from one case.

Measured v12 per-decision encoding time was approximately 0.19 s and planning
17.14/17.29/17.91 s; CEM planning was 17.07/17.28/17.93 s. These are
single-run measurements, not a comprehensive compute comparison.

## Verification and failed attempt

The first v12 attempt was stopped with both arms failed before any planner
model call or planned action dispatch: the enclosing sandbox denied bwrap's
network namespace. Its partial outputs are retained in
`le-wm-main/results/pc_jepa_native_v12_feedback_dev_20261104_codex/`. A new
protocol and new seeds were used after a separate authorized preflight; the
failed attempt was not resumed or overwritten.

The saved-data audit for the second attempt checks version and dependency
hashes, witness, input history, normalized actions, re-evaluated carry,
per-call accounting, pre-pair incumbent, final paired selection, and exact
dispatch bindings. The first stricter audit stopped because it incorrectly
required the final re-evaluation score to equal the earlier batch minimum:
the identical first-decision incumbent had an increase of
`3.725290298461914e-09` in its final comparison batch. The revised audit
separately verifies that the incumbent action was the minimum **before** the
pair and that selection agrees with the recorded final pair; it records the
cross-batch difference instead of asserting its cause. This change corrects
the audit's comparison, not the planner's scientific success criterion.

The saved-data audit passed. A separate process replayed both saved physical
command streams in the same native backend; all 26 RGB, state, proprioception,
public-state and transition boundaries per arm matched exactly. This does not
independently replay the neural predictions or private solver caches.

Protocol: `docs/PC_JEPA_NATIVE_V12_FEASIBLE_FEEDBACK_V2_DEV_PROTOCOL_20261104_CODEX.md`.
Results, audit, journals, actions and physical replay:
`le-wm-main/results/pc_jepa_native_v12_feedback_v2_dev_20261104_codex/`.
