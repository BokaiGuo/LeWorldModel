# PointMaze V6 Counterfactual-Contrast Confirmation

## Motivation

The sealed V6 panels removed V5's outcome leakage, calibration reuse, and
candidate-support confound, but their recurrent predictor is trained primarily
on terminal-effect MSE while the scientific estimand is root-level candidate
regret.  A low average effect error can still flatten or invert the difference
between two candidate actions from the same source root.

The V7 confirmation therefore keeps terminal-effect supervision and adds a
source-root-only task-contrast term.  It is not a new outcome-side tuner: all
contrast labels come from the train-source successor arrays, and the contrast
weight is selected before the sealed outcome roots are simulated.

## Objective

For a source root `r`, candidate `i`, predicted terminal effect
`e_hat(r,i)`, source position `x_r`, goal `g_r`, and observed source task
cost `c*(r,i)`, the added objective is:

```text
L = L_effect + lambda * Huber(
      ||x_r + e_hat(r,i) - g_r|| - ||x_r + e_hat(r,j) - g_r||,
      c*(r,i) - c*(r,j)
    )
```

The second term compares only candidate pairs from the same root.  The runner
rejects a source artifact if those pairs do not share a source state, goal, and
candidate-id bank.

The existing sealed state-interaction source artifact has 64 roots and 64
candidates per root.  Its realised source-only task-cost span is nonzero for
all roots (mean `0.3320`, minimum `0.1140`), with a mean best-versus-second
best oracle margin of `0.03487`.  Its rootwise pair-contrast RMS is `0.10623`.
The runner divides contrast residuals by this train-source-only scale before
applying Huber loss; without that normalization a unit contrast weight would
be dominated by normalized effect MSE.  This establishes that a rootwise
contrast is defined on the training source; it is not evidence that the new
loss improves held-out ranking.

## Frozen choices and safeguards

- Candidate generalization is `shared_candidate_in_support`; this isolates
  history/root transfer from candidate extrapolation.
- The train, selection, authorization, and outcome roots remain disjoint.
- `lambda` is selected only from `{0, 0.25, 1.0}` on selection roots alongside
  hidden width, temperature, and the visual-history routing policy.
- Authorization uses the other deterministic calibration half.  Only a
  positive paired lower confidence bound over action-only permits the learned
  route.
- The score stage loads the hash-bound frozen checkpoint and is the first
  operation allowed to materialize outcome labels.
- A selected `history_energy_gate` is validated as an executable frozen policy;
  it cannot silently fail because of a stale internal policy name.
- The recurrent encoders are manually unrolled CUDA `GRUCell` modules.  This
  preserves causal gated recurrence while avoiding a known system-cuDNN versus
  PyTorch-cuDNN incompatibility; preflight executes that exact CUDA path.

## Required evidence

The CUDA-only launcher is:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
./.venv/bin/python -m experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v7_cuda
```

Its result is confirmatory only if the new immutable output contains:

1. a frozen selection with a hash-bound V6 model and selected contrast weight;
2. separate authorization roots with positive recurrent-or-routed regret gain;
3. outcome labels generated after that frozen artifact; and
4. a positive held-out paired regret interval, with the observability controls
   showing that the candidate bank carries usable root-conditioned signal.

Until then this is a protocol and implementation change, not a positive
world-model or closed-loop planning result.
