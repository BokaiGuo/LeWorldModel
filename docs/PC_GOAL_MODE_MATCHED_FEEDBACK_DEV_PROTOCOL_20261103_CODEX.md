# Goal-mode feedback controls: fixed development protocol

The four previously used PointMaze roots are a development panel, not
independent confirmation. This protocol is written after observing previous
single-root goal-mode and CEM outcomes. It does not reopen the stopped
PointMaze non-immediate mechanism experiment.

Use the same frozen checkpoint, RGB goal, warmup, action normalization and
legacy RGB bridge for each arm. At each of three feedback decisions use an
actual three-frame RGB context and ten most recent executed actions, plan 15
actions, and execute five. Seal each decision before execution. Per decision
enforce at most 132 five-action model-prediction token samples:

* `goal`: response Jacobian, goal-ranked direction and zero, common 12-sample
  suffix CEM per candidate; select complete-chain score with zero-prefix
  abstention threshold 0.
* `greedy`: identical goal bank and suffix search, but select the candidate
  with smallest immediate model prefix cost; do not filter it post hoc.
* `ridge`: same response Jacobian, full ridge response direction and zero;
  same suffix search and abstention rule as `goal`.
* `random`: zero and plus/minus one uniformly seeded random direction. Do not
  compute an unused response Jacobian; allocate the remaining token budget
  to suffix CEM as 15, 14 and 14 samples in fixed bank order.
* `cem`: 44 full-horizon CEM samples, no prefix bank.

Use fixed seeds `2026110800 + 10*root + step` for random directions and
`2026110900 + 10*root + step` for CEM. Each candidate within a decision uses
the same CEM seed. Log full candidates, decisions, actual inputs/actions,
prediction calls and token samples, encoder count/time, structural operations,
wall time and hashes. If a generated direction is degenerate, use the existing
bank without inventing a candidate; unused budget is explicitly reported.
Greedy and goal have the same candidate bank only at identical contexts; their
feedback histories may diverge after execution.

Prediction-token equality is not total-compute equality. These exploratory
results cannot establish PC2/PC3, independent performance or training-set
exclusion; the baseline has not been granted measured structural-cost-equivalent
extra resources. Compare physical outcomes only by separate post-seal replay,
with all arms and roots retained regardless of outcome.
