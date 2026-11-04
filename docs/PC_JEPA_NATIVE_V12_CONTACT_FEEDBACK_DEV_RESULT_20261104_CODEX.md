# v12 native PushT contact-goal feedback: development result

## Feasibility witness

On one fixed new root, the predeclared evaluation-only feedback law produced
15 legal actions that moved the object's centre **44.704 pixels** and rotated
it **0.44737 radians** from the post-warmup state. Replaying these fixed actions
from reset reproduced RGB, proprioception and physical state exactly at all
26 boundaries. The post-warmup state fails the specified target success rule;
the witness terminal state is the constructed target. The feedback law used
physical state and is not a planner or a performance baseline.

## Actual feedback

Both v12 and zero/carry CEM completed all three 15/10/5-step decisions with
1920 prediction tokens each, dispatched their chosen actions, and failed the
physical success rule. Distances below are in native physical coordinates.

| Boundary | v12 agent error | CEM agent error | Object error, both arms |
|---|---:|---:|---:|
| after warmup | 321.679 | 321.679 | 44.704 |
| after 5 actions | 437.563 | 252.824 | 44.719 |
| after 10 actions | 610.755 | 217.918 | 44.720 |
| after 15 actions | 569.954 | 204.487 | 44.720 |

Object errors were effectively unchanged and the final object-angle error was
0.44723 radians for both. v12's model-selected latent MSE decreased across
its three changing observation contexts, while its physical agent distance
increased; those latent values are not a common fixed-context trajectory or
proof of a particular model or scoring defect. The target is known to be
reachable via the evaluator-only witness, but one exposed constructed goal is
not a random-goal feasibility claim or an independent performance test.

Both decision streams passed saved-data checks of model/version hashes,
histories, actual actions, carry, selected-chain binding, token accounting,
incumbent and same-batch pair. A separate same-backend process replayed each
arm's 26 boundaries exactly, including RGB, proprioception, public state and
transitions. The neural predictor was not independently replayed. The
evaluation-only witness and privileged state were not made available to the
isolated worker. No specific planner, model or environment bug is established
by this result.

The task stops under the fixed protocol; no retries or tuning on this goal.
It provides a concrete PC0 development witness and a bounded negative
closed-loop observation, not PC2 mechanism, PC3 advantage, PC4 success or PC5
generalization.

## Artifacts

- Witness protocol and records:
  `docs/PC_JEPA_NATIVE_CONTACT_WITNESS_DEV_PROTOCOL_20261104_CODEX.md` and
  `le-wm-main/results/pc_jepa_native_contact_witness_dev_20261104_codex/`.
- Feedback protocol:
  `docs/PC_JEPA_NATIVE_V12_CONTACT_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md`.
- Both trajectories, full decision records, saved-data audit and physical
  replay:
  `le-wm-main/results/pc_jepa_native_v12_contact_feedback_dev_20261104_codex/`.
