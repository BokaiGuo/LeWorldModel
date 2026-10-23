# PointMaze V23: fixed-bank epsilon-regret decision transfer

Status: `completed on host CUDA; positive matched-transfer direction with a
stratum-specific coverage failure`.

## Why this panel exists

V22 established a precise C3 failure under exact frozen-model action support:
the simultaneous rootwise task-contrast envelope was larger than every
predicted unique-winner margin, while the physical best-versus-second gaps in
the declared 16-action bank were also below that envelope.  A zero-release
unique-winner rule was therefore the expected fail-closed result, not evidence
that lowering a threshold would make planning reliable.

V23 tests a different, predeclared decision object.  On the same fixed-bank
coverage event, a predicted winner has a joint upper regret bound

\[
  \max_{j \ne \hat i}(q_\alpha - (\widehat J_j-\widehat J_{\hat i}))_+.
\]

It is permitted only when this bound is at most the fixed task-cost budget
\(\epsilon=0.12\), action/state support passes, and the independent
calibration risk gate passes.  The stricter unique-winner certificate remains
reported separately.  This is not a relaxation of the conformal envelope.

## Protocol frozen before outcome labels

- New generated root range: `30664–31495`; train/calibration/outcome each use
  256 root-disjoint episodes, plus 64 disjoint candidate-design roots.
- V23 fits one predeclared recurrent, action-conditioned effect model with a
  rootwise task-contrast loss on train roots only.  It has no calibration model
  search.
- The candidate bank is shared exactly between source training and evaluation;
  it is fixed before calibration and outcome scoring.
- Calibration roots fix the contrast envelope and model-only risk bound.  The
  corresponding gate JSON is hash-bound to outcome labels before any outcome
  label is generated.
- The matched fallback uses action-only scores on exactly the roots the
  epsilon-regret rule does not permit.

## Relation to planning-shaped baselines

This is an action-conditioned counterfactual-effect test, not a reproduction
of PlanWAM or a claim about its driving performance.  Planning-shaped/shared
future representations are treated as baselines whose average task score does
not, by itself, establish counterfactual decision sufficiency.  The empirical
question is whether a repaired fixed model can yield more useful certified
fixed-bank decisions on roots never used for training or calibration.

## Required outcome report

The eventual report must give, separately:

1. contrast bound, risk threshold, and simultaneous independent-root coverage;
2. unique-winner release and epsilon-regret release;
3. realized regret, regret-bound validity on covered roots, ranking error, and
   matched-fallback paired regret;
4. action-bank/root disjointness and the gate-to-outcome hash binding.

Neither a positive release rate nor an average regret change alone establishes
permit-conditional safety, online MPC success, contact transfer, or universal
world-model capability.

## Completed outcome audit

The host-CUDA run completed with a root-disjoint 256/256/256 panel and exact
gate-to-outcome hash binding. The frozen global contrast bound was `0.04841`.
The predeclared \(\epsilon=0.12\) rule released `239/256` roots, while the
strict unique-winner rule would have released only `23/256`.

Against the exact release-mask action-only fallback, the paired outcome regret
gain was `+0.08933` (paired bootstrap 95% interval `[+0.06973, +0.10954]`)
and success gain was `+0.21875`. This is a real independent positive direction
for the fixed V23 protocol, not a post-hoc threshold choice.

However, observed simultaneous contrast coverage was `0.85156`, below the
nominal marginal target `0.90` (Wilson 95% interval `[0.80282, 0.88991]`).
It would be incorrect to present V23 as a validated 90% coverage result.

The post-outcome mechanism audit localizes the shortfall:

| Observable stratum | Outcome coverage | Mean matched regret gain | Calibration roots exceeding global bound |
| --- | ---: | ---: | ---: |
| visible high motion | 0.95313 | +0.17715 | 0.00781 |
| direct zero motion | 0.75000 | +0.00152 | 0.17188 |

Thus the global mixture envelope is not transport-stable across the two
predeclared observable regimes. V24 must use **new roots** and predeclare both
(i) recurrent repair only for the visible-motion stratum with action-only
fallback for direct-zero-motion roots, and (ii) stratum-specific calibration.
V23's opened calibration/outcome data may not choose V24's thresholds,
epsilon, or model width.
