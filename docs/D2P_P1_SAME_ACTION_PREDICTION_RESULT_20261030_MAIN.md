# P1 same-action prediction/RGB result

## Scope

This record reports the host-CUDA execution of
`codex_four_arm_same_action_prediction_20261030_v1`. It reconstructs the
already sealed four-arm common-RGB trajectories and, at boundaries 0, 5, 10,
and 15, predicts the arm's own executed five-step action packet from the
saved context. It then encodes the matching physical RGB checkpoint at the
next boundary. This is a post-hoc known-root diagnostic, not an independent
confirmation, equal-budget comparison, or P1 admission.

Neural inference ran on host CUDA only. The frozen model hash before and after
the run was unchanged. The result has 16 action-encoder calls, 16 prediction
calls, and 16 RGB-encoder calls.

## Immutable receipts

- Output: `le-wm-main/results/codex_four_arm_same_action_prediction_20261030_v1_HOST01/`
- `result.json` SHA-256:
  `ec7fd4edff5c673b8f4e42ddc66e88641d1616670fcda1b15efa2211a8f3ff51`
- `journal.jsonl` SHA-256:
  `d58ccb154f11f95ff1617f63a89d82b0f670cdef1492f3a30e37bfb37b779a25`
- Status: `exploratory_complete`
- P1 admission: `false`
- Independent confirmation: `false`

## Saved result summary

| arm | mean prediction MSE | mean predicted-goal MSE minus actual-goal MSE |
| --- | ---: | ---: |
| structured | 0.030745 | -0.007532 |
| native | 0.042976 | +0.040596 |
| random | 0.019932 | -0.013070 |
| immediate_progress | 0.047205 | +0.046078 |

The four-arm aggregate prediction MSE is `0.035215`.

## Interpretation boundary

The residual is not a constant global offset: its magnitude and signed
goal-error difference vary by arm and boundary. The structured and random arms
have slightly lower mean predicted-goal MSE than the corresponding actual-goal
MSE, while native and immediate-progress have higher values. This is evidence
for trajectory/context-dependent model-to-RGB mismatch, but it does not
identify whether the cause is model rollout error, task-metric mismatch, or
candidate-generation quality.

The experiment does not establish a causal capability change because each
packet is evaluated on its own previously executed trajectory. It therefore
cannot open P1 or replace the fixed three-prefix H10 intervention. The next
valid join is the complete H10 saved consumer and analysis, with this result
used only as a same-action prediction diagnostic.
