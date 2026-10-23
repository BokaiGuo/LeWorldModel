# PointMaze V11: pairwise counterfactual-identifiability protocol

## Scientific role

V11 tests C1 only: whether a model can infer which member of a predeclared,
same-summary action pair has lower realised task cost from a causal visual
history.  It is not a 16-way planner confirmation, a closed-loop result, or a
repairability claim.  The primary held-out quantities are pairwise sign
accuracy, contrast MAE, and coverage above the frozen realised-cost margin.

This scope follows a concrete V10 finding.  A full candidate-bank action
choice may have negligible incremental history signal even when individual
state-conditioned action pairs exhibit preference reversals.  Treating the
latter as evidence for 16-way planning would be an estimand error, not a
positive planning result.

## Why the first V11 preflight was rejected

The initial V11 construction used
`ultra_tail_robust_goal_balanced_interaction_pairs`.  That bank was selected
for the full-bank best-versus-second task-cost gap, while V11 was evaluated on
adjacent, summary-collision pairs.  On V11's fresh design roots, all static
checks except the pairwise lower-tail margin passed: the pooled q10 absolute
pair contrast was below `0.003`.  No model panel, calibration labels, outcome
labels, or fitted model were produced.

The response is not to lower the margin.  It is to make candidate construction
match the declared estimand.

## Repaired candidate contract

`pairwise_contrast_robust_interaction_pairs` generates 1,536 deterministic
interaction-pair proposals and evaluates each proposed pair on two disjoint
design-root folds.  A selected pair is ranked by:

1. state-conditional sign reversal across design roots;
2. its minimum-fold q10 absolute realised-cost contrast;
3. its minimum-fold fraction exceeding the fixed `0.003` margin;
4. sign balance, mean contrast, and a deterministic proposal-index tie break.

This does not read train, calibration, or outcome roots.  It preserves the
shared in-support action bank: the held pair actions are also present in the
source scorer's action support.

The static preflight authorizes the CUDA build only when all of the following
are true:

- evaluation and design roots are disjoint from each other and all prior
  contexts;
- exact held actions are contained in source training support and are fresh
  relative to prior contexts;
- the pooled q10 absolute contrast is at least `0.003`;
- at least 80% of pair-root contrasts meet that margin; and
- all eight pairs have state-conditionally mixed preference signs.

## Execution boundary

The host-only entrypoint is
`experiments/decision_benchmark/run_pointmaze_pairwise_contrast_v11_host.sh`.
It requires visible `/dev/nvidia*`, a CUDA allocation, and a manually unrolled
GPU `GRUCell` preflight.  The actual fitted model never falls back to CPU.

Only after the static contract passes does the runner build frozen visual
histories and fit the recurrent/action-only controls on source roots.  Model
width, contrast weight, temperature, and history representation are selected
on one deterministic calibration-root half by pairwise sign accuracy (then
contrast MAE), not by full-bank regret.  The other calibration-root half
authorizes release only if a root-clustered bootstrap lower interval for the
pairwise sign-accuracy gain over action-only is positive.  Only then are
held-out outcome labels materialized.  Full-bank regret is recorded only as a
secondary diagnostic.  A pairwise improvement
does not justify a C2 repair, C3 certified planning, or RGB-only deployment
claim without separate evidence.
