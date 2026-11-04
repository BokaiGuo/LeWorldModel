# Residual-spectral prefixes: cp5 feedback experiment

This is a known dev303 root experiment, not an independent confirmation.
The new feedback entry point binds candidate 4 (spectral_0_mode2_+1,
selected before physical execution by latent scoring), 18 (zero), 19
(task_greedy_direct), and 20 (random_control). Each uses its own actual
cp5 RGB and action history. Outcomes and simulator state do not enter the
planner. Two objectives, latent and goal-tangent, are kept separate.

## Execution and verification

The eight CUDA plans completed with an unchanged frozen model. Each plan
used 10 action-encoder and 10 predictor calls, each covering 3200 samples.
Shared preparation used 11 goal encodes and four cp5 encodes. These matched
suffix budgets do not certify equal total method cost, including prefix search.

All eight physical branches replayed the exact five-step prefix and then
executed ten newly planned actions. None terminated or truncated. Saved
verification checked 144 frame references, full action traces, prefix state,
snapshot and RGB replay, report shards, source closure and journal hashes.

| Prefix | Objective | Terminal reward | Object position error | Angle error |
|---|---|---:|---:|---:|
| spectral | latent | -369.291 | 276.055 | 2.247 |
| zero | latent | -359.818 | 236.638 | 3.051 |
| greedy | latent | -291.535 | 246.341 | 2.783 |
| random | latent | -357.582 | 236.638 | 3.051 |
| spectral | goal-tangent | -307.027 | 237.466 | 3.028 |
| zero | goal-tangent | -352.953 | 236.638 | 3.051 |
| greedy | goal-tangent | -285.198 | 239.424 | 3.080 |
| random | goal-tangent | -382.190 | 236.638 | 3.051 |

This is descriptive, with dependent arms on one development root. RGB
refresh and a common suffix optimizer did not establish a spectral-prefix
advantage: its reward remained below greedy for both objectives. The
goal-tangent spectral arm has slightly lower object position error than
greedy, illustrating why reward and pose must not be conflated. No metric
was changed after outcomes. The prefix already improved pusher position,
so this is not evidence of non-immediate progress under an overall task metric.
Common-suffix intervention, independent roots and full strong-baseline
comparison remain missing. P1 admission remains false.

## Artifacts

- CUDA report: `le-wm-main/results/p1_residual_cp5_feedback_20261102_RUN_DEV01/report.json`
  SHA256 `afc7c7550ad8b72dfa5f2447dc5a190a4076eefef6c037c101e4d99aa0964405`
- Physical report: `le-wm-main/results/p1_residual_cp5_physical_20261102_RUN_DEV01/report.json`
  SHA256 `c66b42a3ab056b5a422515ae6932b6455d7dc994f6771d86861ac6af323b7191`
- Saved analysis: `le-wm-main/results/p1_residual_cp5_saved_20261102_DEV01.json`
  SHA256 `123652f6ab725c1a2235e39fad10b7316feb3c7f8ebd69903aa375d876911e11`

The code-debugging workflow was used to check input bindings, time alignment,
normalization and complete execution before interpreting the negative result.
This does not exhaust all possible model-interface or objective errors.
