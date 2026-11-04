# P1 Reacher: common-tail tangent development protocol

This is an outcome-informed **development hypothesis**, not an independent
confirmation or a change to any gate. The two roots and some predecessor
outcomes have already been exposed. No physical outcome is an input to the
proposal code or source preflight.

## Motivation and distinction

The goal-Q physical report has no non-immediate witness on root 20263160 and
an ineligible zero fallback on 20263161. Existing prefix refinement optimizes
five one-token probes, existing goal-Q selection scores each candidate's own
previous CEM suffix, and existing physical common-tail work uses exposed
actions. The new method generates candidates using the *same five fixed
two-token suffix parameter arrays* for every prefix, with the same goal-only
Q score for immediate and final latents. It optimizes a tangent direction of
mean active-suffix cost relative to the neutral-parameter suffix, then adds
the same small immediate-cost-increasing normal component to both signs.

This tests whether model predictions distinguish suffix *response*, not just
an unconditional final-state advantage. It is not a measurement of actual
physical suffix capability. A zero parameter is passed through the existing
bounded action map; it is **not asserted to be zero physical torque**.

## Fixed proposal and scoring

For each root use the zero 10-dimensional prefix parameter and five fixed
tails: neutral and signed 0.3 parameter values on each of two action
coordinates, repeated across five action steps and both tail tokens. The
first tail is the neutral-parameter control. Compute the goal-Q immediate
gradient and the gradient of mean active-tail-minus-neutral-tail final cost.
Project the latter's negative gradient off the immediate-cost gradient; fail
closed if either required direction is degenerate. Check its model directional
derivative by a symmetric 0.001 finite difference with relative tolerance 0.20.

Evaluate the neutral prefix plus six proposals, in radius order 0.05, 0.10,
0.20 and positive/negative tangent order. Directions include a 0.25 normal
component before normalization. Keep all seven rows including rejected ones.
Relative to the neutral prefix, the model-only diagnostic selector requires:

1. immediate Q cost at least `1e-6` higher;
2. mean final Q difference at most `-1e-6`, and no common tail worse;
3. mean active-tail difference adjusted by the neutral-tail difference at
   most `-1e-6`, and no active adjusted difference worse.

Ties in total same-tail final difference use proposal index. If none qualify,
return `None` with no zero fallback masquerading as selection. These are
candidate-generation rules, **not** a revision of the physical P1 gate.
Degenerate gradients or a failed finite-difference check are failures, not
permission to tune radii, threshold, root or outcome.

## Executable CPU preflight

From `le-wm-main`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_reacher_pc_p1_common_tail_tangent_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_preflight_20261102_codexexclusive.py tests/test_reacher_pc_goal_q_selection.py tests/test_reacher_pc_feasible_prefix_backtrack.py tests/test_reacher_pc_spectral_candidates.py
```

To reproduce source preflight, choose a **new** output name:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.reacher_pc_p1_common_tail_preflight_20261102_codexexclusive --output /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_03.json
```

Preflight checks source report, existing Q receipt and its FD checks,
checkpoint and action-statistic file hashes, fixed root/history identities,
and proposal-code hash. It does not instantiate the checkpoint, reconstruct
the roots, verify host CUDA, or authorize model or physical execution. The
first preflight receipt `_01` references an earlier proposal-code version;
`_02` references the final version. Neither is a physical result.

## Next coordinated model-only run

An executable end-to-end host launcher is **not yet implemented**. Before
running one, separately coordinate host CUDA and implement an exclusive
launcher/consumer that:

- seals the preflight, this protocol, code, environment and checkpoint;
  verifies the frozen loaded parameter hash before and after;
- reconstructs the fixed two seed/history contexts and goal RGB with the
  existing model-only runner's source mechanism; validates root and goal
  hashes and raw/normalized action chart before prediction;
- reconstructs the existing goal-only Q from the sealed per-root Gram and
  validates its source and FD receipt; preserves all seven proposals, all
  per-tail scores, both derivative checks, failures and no-selection states;
- performs one adapter call per root, with no physical outcome input and no
  retry or alternate hyperparameters. The logical nominal count per root is
  9 prefix-predict samples, 90 suffix-token predict samples and 2 prefix VJP
  calls; account separately for goal encoding/Q, context encodes, module
  calls, memory and wall time. These counts do not establish equal total cost;
- freezes parameter-space and actual bounded/raw action arrays, their
  float32 bytes, identities and all model selection outputs before any new
  physical outcome is opened.

Only after a separately frozen CPU physical protocol, compare **all seven
prefixes against all five identical tails** on each root (up to 70 branches,
15 physical steps each, plus root reconstruction/warmup). Preserve duplicates,
guard failures, terminations and all denominators. Evaluate the same-tail and
neutral-tail-adjusted physical changes; if the runner lacks a native success
receipt, mark success unknown. Include task-specific strong baselines and
charge their complete model and physical budgets. The proposed common-tail
matrix is diagnostic, not a substitute for equal-budget closed-loop suffix
planning. Fix actual resource caps and output seals before dispatch.

Neither the new Q-model criterion nor any result on these known roots proves
physical non-immediate reconfiguration, independent generalization, native
task success, equal-budget benefit, or P1 admission.
