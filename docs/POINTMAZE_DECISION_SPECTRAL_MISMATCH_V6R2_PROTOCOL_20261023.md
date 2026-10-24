# PointMaze Decision-Spectral Mismatch V6R2: frozen P1 protocol and result

## Question

Before attempting a spectral repair, does a task-output, action-support-aware
real-versus-predicted finite-horizon response mismatch explain held-out
endpoint-ranking regret?

## Frozen construction

- Roots: 128 train, 192 calibration, and 64 outcome roots from the new
  `36552..37191` range.  No root or candidate-action hash overlaps a supplied
  historical context.
- Candidate support: 16 five-pulse, 20-step action sequences.  The held bank
  is exactly included in the 64-sequence training bank; this is a
  root/history-generalization control, not continuous-action evidence.
- Task output and metric: realized terminal PointMaze `qpos=[x,y]`, with the
  identity metric in that physical output coordinate system.
- Per root: center the flattened candidate sequences, retain only their SVD
  support directions, fail on no action variation or an ill-conditioned
  retained support, fit real and predicted endpoint responses separately on
  that same support, then compute
  `E = Q^(1/2) (G_pred - G_real) Sigma_C^(1/2)`.
- The candidate pair used for the local leverage diagnostic is the frozen
  endpoint choice versus frozen cost-only choice.  It is selected from
  predictions, not realized terminals.
- Before any calibration or outcome terminal is materialized, the runner
  writes `pre_outcome_predictions.json`.  The final report binds its SHA-256;
  the P1 audit rejects a missing or altered bundle.
- Inference uses roots, not candidate pairs, as independent units.  The P1
  association is Spearman mismatch spectral norm versus endpoint regret,
  evaluated separately on calibration and outcome roots.  No threshold,
  mode, or policy is selected from those associations.

## Execution evidence

- CUDA-only launcher:
  `le-wm-main/experiments/gdd_wm/run_pointmaze_counterfactual_endpoint_v6r2_decision_spectral_cuda.sh`
- Endpoint report:
  `le-wm-main/results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_endpoint_20261023_cuda/report.json`
- Contract audit: passed all static protocol checks, including root isolation,
  deferred outcomes, ordered-action contracts, and pre-outcome aliases.
- Sealed prediction-bundle SHA-256:
  `1a6220d58f98ab6dc25264678f7ba311f595d769df6668b928ffbd9d8f1d072c`.

## Result and decision

The root-level held-out association is negative and imprecise:

| split | roots | Spearman | bootstrap 95% interval | two-sided permutation p |
| --- | ---: | ---: | --- | ---: |
| calibration | 192 | -0.0877 | [-0.2262, 0.0552] | 0.2227 |
| outcome | 64 | -0.1274 | [-0.3812, 0.1204] | 0.3238 |

The V6R2 endpoint arm is descriptively lower-regret on outcome roots
(`0.03129` versus cost-only `0.03802`), but its independent certification
upper bound is `+0.001318`, so the frozen policy correctly deploys cost-only.

**P1 decision: no-go for the current mismatch metric as a predictive
mechanism.**  Do not add more variants of this spectral score or use it to
choose repair modes.  This result does not rule out all decision-spectral
mechanisms: it is one matched fixed-bank PointMaze protocol, not continuous
actions, RGB-only MPC, a repair intervention, or a universal negative result.

## Downstream stage closure

The CUDA P2 synthetic formula calibration passed: task-output mismatch tracked
candidate regret (Spearman `0.9241`, permutation `p=5e-5`), generic latent
endpoint RMSE did not (`0.0342`, `p=0.8182`), and the task-output gauge check
was numerically invariant.  This validates the implementation only.

The CUDA P3 rank-matched synthetic repair stress test is a no-go for the
method claim.  Spectral rank-1 repair improved on random rank-1 and the chosen
PCA-axis control, but ordinary full action-effect learning had lower realized
regret (`0.8088` versus `0.9628`).  Therefore P4 recalibration, P5 RGB MPC,
P6 PushT transfer, and P7 external transfer are formally blocked: they would
otherwise transfer an unvalidated repair mechanism.  The machine-readable
decision is `le-wm-main/results/decision_spectral_stage_gates_20261023_cuda.json`.
