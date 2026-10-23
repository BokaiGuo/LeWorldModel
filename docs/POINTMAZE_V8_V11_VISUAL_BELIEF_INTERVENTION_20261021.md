# PointMaze V8–V11: visual belief intervention

## Status

Completed on CUDA (RTX 5060 Ti) with root/action-hash isolation.  This is a
mechanism study, not a claim that frozen RGB planning is now solved.

## Question

The sealed V7 state ceiling showed that the 20-step candidate bank was
rankable from privileged `[x, y, vx, vy]`, while direct visual ranking did not
pass its authorization gate.  V8–V11 tested whether causal visual history can
first recover that control state and then recover task-conditioned action
ranking.

The runtime route is:

```text
eight causal RGB latents + seven prior action blocks
  -> visual belief [x, y, vx, vy]
  -> pulse-aware candidate task ranker
```

Simulator state is used only as a source-training target and post-hoc held-out
readout.  It is never a runtime input.

## Protocol repairs

- V8 uses a new 2,640-root pool and excludes every opened V1–V7 root and
  candidate-action identity.
- V9 fixes the direct visual ranker before outcomes; it removes online family
  selection and retains an independent authorization gate.
- V10 increases the root budget to `128 train / 160 calibration / 64 outcome`
  after locating high authorization variance in the earlier 24-root gate.
- V11 keeps the V10 budget but fixes the source-supervised visual-belief arm.
  It uses new roots and new candidate action hashes; the builder rejects any
  root or action-hash reuse.

## Sealed results

| artifact | authorization | held-out task-ranking result | interpretation |
| --- | --- | --- | --- |
| V8 belief/direct development panel | selected action-only; not deployable | belief gain `+0.02685`, direct gain `+0.03784` over action-only | useful direction only; selector variance prevents a controller claim |
| V9 fixed direct | fails: CI `[-0.00730, +0.02513]` | mean direct gain `+0.01544`, CI `[-0.01296, +0.04399]` | no confirmation |
| V10 fixed direct, expanded gate | fails: CI `[-0.00701, +0.01528]` | mean direct gain `-0.00820`, CI `[-0.02486, +0.00613]` | refutes stable direct-ranker repair |
| V11 fixed belief, expanded gate | passes: gain `+0.01432`, CI `[+0.00029, +0.02954]` | mean belief gain `+0.00030`, CI `[-0.01392, +0.01314]` | state-belief route is authorized but does not show stable held-out task-transfer gain |

The V11 matched-history belief ranker does beat the cyclic-history control by
`+0.02116` regret gain, CI `[+0.00189, +0.04058]`.  Its state readout also
improves from cyclic to matched history: state RMSE `1.780 -> 1.475`, velocity
sign accuracy `0.531 -> 0.625`.  Hence the negative downstream result is not
evidence that causal history is unused.  It is evidence that the recovered
control state is not yet transfer-sufficient for the current global task-cost
ranking objective.

## Conclusion and boundary

The experiment rules out two overclaims:

1. a direct frozen visual task ranker is a stable repair; and
2. improving visual control-state observability automatically improves
   candidate action ranking.

It supports the narrower causal decomposition:

```text
visual history -> control-state observability       supported
control-state observability -> global action ranking not yet supported
```

The appropriate next intervention is therefore not another global history
encoder.  It must test a local, observed-effect / task-conditioned update
between recovered belief and candidate ranking, with a fresh predeclared
panel and the same authorization rule.

## Artifacts

- `le-wm-main/results/pointmaze_v8_visual_belief_state_20261021_cuda/`
- `le-wm-main/results/pointmaze_v9_fixed_direct_confirmation_20261021_cuda/`
- `le-wm-main/results/pointmaze_v10_expanded_calibration_confirmation_20261021_cuda/`
- `le-wm-main/results/pointmaze_v11_fixed_belief_confirmation_20261021_cuda/`
