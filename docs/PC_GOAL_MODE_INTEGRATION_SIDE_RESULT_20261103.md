# Goal-relevant mode compiler: isolated development result

## Implemented scope

The opt-in `TaskModeCompiler` uses the existing Atlas multi-layer search,
bounded nonlinear program scoring, and complete executable action witness.
It selects nonzero modes either by task-related damped objective reduction
or by singular-value order. Progress, bridge directions and local ridge
value use the same retained modes. The generic feedback plan adapter accepts
its complete selected plan. This verifies an interface, not real RGB feedback.
Shared default compilers and historical gates were not modified.

## Completed checks and experiment

The existing response-program joint-refinement artifact was checked against
protocol and source hashes. All 192 selected chains were independently
replayed, their budget arithmetic checked, and response-arm final-chain
interventions recomputed. No distance discrepancy exceeded 1e-12.
The saved experiment has zero limited mechanism witnesses at each budget.
It was not rerun or reclassified as independent evidence.

The side protocol was then executed on all 16 predeclared exposed exact
unicycle goals:

| Quantity | Result |
| --- | --- |
| Gain-ranked mean terminal distance | 0.2893616484512659 |
| Task-ranked mean terminal distance | 0.2893616484512659 |
| Identical selected actions | 16 / 16 |
| Observed task-response ranks | 1 only |
| Total geometry/SVD calls | 224 |

With only one nonzero mode, these contexts offer no alternative mode to
rank. Therefore this panel cannot test a benefit of task-aware truncation.
The rank-two numerical fixture verifies that an irrelevant large-gain mode
can be rejected in favor of a task-aligned smaller mode; that fixture is not
empirical planning evidence. Full-mode agreement with weighted ridge is also
tested and does not establish SVD-specific benefit.

Saved records include source hashes, programs, selected and qualified
indices, mode contributions, actual backend action steps, operation counts,
and planning time. This does not certify neural encoding, GPU or end-to-end
resource fairness. No visual model or physical environment was executed.

## Next admission boundary

Do not promote this implementation on the strength of these results.
A new, prospectively specified development experiment needs nonzero
multi-step action nominals with a verified rank-two task response to test
mode selection. Compare task-truncated, gain-truncated, full-response
no-SVD ridge, and random initialization under a common optimizer and
declared resource policy. Preserve the rank-one panel unchanged.

Before a visual confirmation, bind the adapter to actual RGB and action
history, meter the entire planner, certify source independence, and test
the same selected chain with common-suffix and independently replanned
suffix interventions. Neither this artifact nor the prior joint-refinement
artifact opens P1 or any independent-confirmation gate.

Artifacts:

- `le-wm-main/experiments/spectral_reachability/pc_goal_mode_integration_side_20261103.py`
- `le-wm-main/tests/test_pc_goal_mode_integration_side_20261103.py`
- `le-wm-main/results/pc_goal_mode_integration_side_20261103.json`
- `docs/PC_GOAL_MODE_INTEGRATION_SIDE_PROTOCOL_20261103.md`
