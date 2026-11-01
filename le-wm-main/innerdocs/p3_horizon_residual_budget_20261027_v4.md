# Fresh finite-horizon residual scheduling v4

Implementation-thread predeclared under authorized implementation, before any
new neural/physical outcome. Parameters below were not personally selected by
the user. Four existing source roots, not new independent evidence. P1/P3 false.

## Why this is not the old experiment

v3 inherited an H1 192x10 derivative with unknown acquisition cost and selected
only fixed-H3 primitive-bank trees. Its saved-array budget-frontier consumer does
not execute this protocol. v4 acquires fresh joint H2 192x20 and H3 192x30 terminal
Jacobians at zero logits, preserving differentiation through the complete cached
context recurrence. It schedules mixed H2/H3 candidates, not v3 beam widths.
It does not change or replace v3 or its seals.

## Frozen algorithm

Same four root/context, goal_vectors[0], Q and pretrained model from validated
v3 preparation. Only context/goal/Q are projected into new packets: neither
inherited derivatives nor prior candidate/physics outcomes drive v4.

For H=2,3 compute a separate zero nominal and vectorized action Jacobian.
With r=Q(goal0-nominalH), J=d terminal/d joint time-major logits, use
chiH=||r-Proj_col(QJ)r||²/||r||². Relative SVD cutoff 1e-6.
Zero residual gives chi=0; zero rank gives chi=1 for nonzero residual.
No suffix detach or prefix recompilation. Each horizon pays one SVD.

Ten six-predict blocks per arm. An H2 block buys three H2 candidates, an H3
block buys two H3 candidates. Uniform gets five blocks each. Dynamic gets
floor(10*(.25+.5*max(chi2-chi3,0)/max(chi2,1e-12))+.5) H3 blocks;
the rest are H2. The formula deliberately tests whether the longer action
image reduces normalized residual, not just H1 drift. This is a local linear
heuristic, not a theorem about long-horizon controllability or physical quality.

Both arms use identical deterministic pool prefixes at each horizon: zero,
then independent Gaussian unit directions with radius R*U^(1/(10H)),
R=sqrt(5H)*.75, PCG64 seed 20261027+100*root_index+H. These are scheduling
proposals, not an SRA baseline. All 10H logit coordinates are free; tanh gives
raw [-1,1]. A joint logit ball is not per-packet balls. Float32 CUDA transition
matches existing adapter. Nonlinear original Q cost selects among both horizons;
ties use H then pool index. Thus selected physical durations can differ (10/15
commands); no fixed-horizon or equivalent search-distribution claim.

Whole winner and independently segmented winner replay must match the saved
complete three-latent/two-action context exactly. No physics or feedback here.
Failed candidates/AD/replays abort without replacement, padding or budget refund.
All attempts/completions go to fsync JSONL; report preserves incomplete roots
and blocks unattempted arms. Partial records are not deployable selections.

## Explicit budget, not total-resource matching

Per root shared: nominal2+ADforward2+nominal3+ADforward3=10 predict and ten
action-encoder entries; two Jacobian API calls, 384 mathematical scalar-output
VJP directions, two SVD calls. VJP directions are not engine backward API counts.
Per arm: 60 search predicts plus reserve6 for double replay (actual4 if H2,
6 if H3). Attributed standalone cap76 includes shared10; adding these caps
double-counts shared work. Unique paired root cap142 predict +142 action encoder
=284 neural forward entries. Four roots <=568 each,1136 entries,1704 action
context tokens,8 Jacobians,8 SVD,1536 VJP directions. Batch1 only, three context
tokens per neural entry; RGB encoder forbidden. Pre-dispatch guard applies to
both entry types including ADforward, and retains failed attempts.

Uniform and dynamic share the same compilation even though uniform ignores
the signal. Shared AD/backward/SVD costs are not doubled or charged as zero.
Additional CPU proposal generation, allocation, candidate comparisons, source
verification and JSON serialization are not neural calls and are included in
wall time, not converted to guessed FLOPs. Completed searches explicitly record
allocation calls, pool calls/random proposals, Q evaluations and n-1 winner
key comparisons. Exporter verification adds another saved-candidate cost/selection
scan per arm; those arithmetic checks are not folded into planner runtime.
Cached root/RGB/goal/Q acquisition
cost is inherited unknown and excluded from this cached-input runtime cap.
Checkpoint loading/setup are timed but not forward calls. No complete historical
pipeline matching, total-FLOP matching, efficacy, P1 or P3 admission claim.

## Execution

Only prepare/verify and array/metadata tests are authorized in this workstream.
Main reviews source and runs HOST CUDA explicitly. No CPU neural fallback,
simulator, download, training, tuning, outcome-root selection or figure creation.
The experiment-code skill is scoped accordingly; no performance-based tuning.

From le-wm-main use `.venv/bin/python -m pytest -q
tests/test_p3_horizon_residual_budget_20261027_v4.py`, then module
`experiments.spectral_reachability.p3_horizon_residual_budget_20261027_v4`
with `--prepare --output NEW_PATH`. Verify with `--verify --prepared PATH
--prepare-sha256 RECEIPT_BYTE_SHA`. Reviewed main execution uses `--execute-cuda`
instead of verify and requires another exclusive `--output NEW_CUDA_PATH`.
Prepare seals transitive source, its own tests/doc, original fixed source
preparation and frozen input bindings. Execution rechecks model bytes and sources.
No post-prepare source edits; changes require a new preparation path/version.

`selected_actions(NEW_CUDA_PATH)` is a read-only eight-arm exporter. It verifies
source preparation, complete root/mode census, both saved-array horizon signals,
allocation, pool actions, original Q costs, scoped actual call counts, stable
winner and both stored complete-context witnesses. It recomputes eight array
SVDs (explicit exporter overhead, no neural calls), and rejects partial reports.
Each selection binds report bytes, protocol ID, root packet and all raw commands.
Future physics must preserve H2/H3 duration strata rather than label all as H3.
