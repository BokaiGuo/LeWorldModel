# Candidate suffix quality diagnosis

Date: 2026-11-04

This is a fixed-opportunity posthoc diagnosis of the sealed spectral,
random-direction, and diversity recursive arms. It is not a PC2 or PC3
confirmation experiment.

## Audit status

- 48 root/arm records were audited (16 roots × 3 arms).
- 2,476 depth-0/1 candidates were replayed.
- Every candidate journal passed action, result, token, retention-binding, and
  deterministic replay checks.
- The audit required two engineering repairs:
  - compare the journal call count with `len(audit_journal()["calls"])`;
  - compare replay inputs after the journal's declared `float32` conversion.
- The original generation manifest was not overwritten. The audit records the
  changed audit-script hash and verifies every other sealed dependency.

The resulting audit is therefore a same-environment deterministic replay, not
an independent replication.

## Main observations

Each candidate received one initial full-chain score plus 120 CEM samples under
the same zero-tail initialization, four-knot optimizer, and candidate-derived
random seed. The candidate pool was not changed and no privileged witness was
inserted.

- 96 candidates were retained and 2,380 were pruned.
- Exactly one candidate reached the diagnostic success threshold
  (`best_score < 0.05`).
- That candidate was pruned: root 1, diversity arm, depth 1, candidate 32.
  Its initial and immediate score were `1.1691071727`; its optimized score was
  `0.0314106130`.
- No retained candidate reached the threshold.
- In the 96 root/arm/depth strata containing both retained and pruned
  candidates, the best pruned candidate was lower in 85 strata and the best
  retained candidate was lower in 11. This is a descriptive diagnostic:
  group sizes are unequal, retention is selected by the original planner, and
  the four-knot optimizer has limited detection power.

The result identifies a concrete possible retention loss: at least one
candidate that was pruned had a good tail under the fixed opportunity. It does
not identify whether the loss is caused by candidate generation, the
pre-screen, the retention rule, or insufficient tail optimization. The single
success and the frequent optimized-score floor near `0.5` are insufficient to
estimate a stable success-probability improvement.

## Consequence for the next experiment

No default search change is promoted from this diagnosis. In particular, it
does not justify adding a final-layer probe, increasing the beam, or extending
the 15-step task.

The next identifiable experiment should keep the same candidate bank and the
same total retention slots, then compare predeclared retention rules under a
common suffix optimizer:

1. original immediate-score retention;
2. response-value retention;
3. state-diversity retention;
4. one greedy slot plus one prespecified alternative slot.

The comparison must report candidate-level suffix quality, complete-chain
quality, all probe and optimization tokens, and the number of candidates
actually evaluated. A fixed-opportunity component diagnosis and a full
cost-matched planner comparison must remain separate. If no rule produces
repeatable improvement across the predeclared development roots, the result
is a bounded negative conclusion about this retention mechanism, not evidence
against the underlying theory.
