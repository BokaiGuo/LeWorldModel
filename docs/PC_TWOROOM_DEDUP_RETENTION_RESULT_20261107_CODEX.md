# Exact-prefix deduplication: development result

On the same 40 exposed RGB TwoRoom roots, exact elementwise float32 action
identity is resolved before model evaluation. First occurrence represents
each equivalence class; fewer than two distinct prefixes is an error. All
three methods receive the same distinct candidate pool. The old experiment
and its results remain unchanged.

| Arm | Selected-chain success | B-useful retained slots | Same selected index as old plan |
| --- | ---: | ---: | ---: |
| Lookahead | 40/40 | 80/80 | 39/40 |
| Descriptor | 40/40 | 78/80 | 39/40 |
| Shuffled descriptor | 40/40 | 76/80 | 37/40 |

Distinct prefixes average 11.2 per root. The combined three-arm run evaluated
3,200 full chains, versus 3,560 for the previous shared-A two-stage run.
The reduction is exactly 360 A-chain queries; the three independent B stages
remain charged. Synchronised score-stage time was 2.23 seconds for the complete
40-root run, excluding model load, environment, preprocessing outside the
score calls, and full-task feedback. Timings are not a controlled repeated
wall-clock benchmark and do not establish a system-level efficiency advantage.

GPU decisions were sealed before the old physical result was opened. Outcome
assessment reused the prior replay after checking root, all action arrays,
model and normalizer fingerprints, generation code and physical artifact
hashes. No new physical execution was performed. Different candidate
batching and a shuffle over distinct candidates confound direct causal
attribution to deduplication. Four targeted tests passed, covering identity,
query scope, fail-closed cases, and the previous two-stage query checks.

This fixes a genuine slot-accounting issue for this new development
implementation. It does not establish a descriptor advantage, generalize
to near-duplicate but unequal actions, prove a mechanism, or change any
PC0–PC5 gate. The saturated pool remains unsuitable for confirming the
descriptor.

[Protocol](PC_TWOROOM_DEDUP_RETENTION_PROTOCOL_20261107_CODEX.md) ·
[implementation](../le-wm-main/experiments/pc_tworoom_dedup_retention_20261107_codex.py) ·
[sealed plan](../le-wm-main/results/pc_tworoom_dedup_retention_plan_20261107_codex.json) ·
[replay comparison](../le-wm-main/results/pc_tworoom_dedup_retention_result_20261107_codex.json)
