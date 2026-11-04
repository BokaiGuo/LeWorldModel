# Feasible-prefix same-action prediction replay

The fixed protocol `PC_REACHER_FEASIBLE_SAME_ACTION_REPLAY_PROTOCOL_20261102.md`
was run on host CUDA for all five arms of both known roots. All 30
arm/checkpoints were recorded. The saved raw and normalized actions decoded
within 1e-6; root/goal RGB hashes, first-successor RGB, and step-5 and step-15
qpos distances met the existing runner's identity checks. The checkpoint
parameter hash was unchanged. The runner reconstructs history from the
recorded seed but does not separately compare the stored history index or
save every full history array, so those stronger bindings remain uncertified.

At each time, the following counts compare all ten unordered arm pairs at
that root. A reversal means the two signed cost differences have strictly
opposite signs; no near-tie tolerance is applied.

| Root | Step | Open prediction vs actual RGB latent | Actual RGB latent vs qpos distance |
|---|---:|---:|---:|
| 20263160 | 5 | 4/10 | 5/10 |
| 20263160 | 10 | 0/10 | 0/10 |
| 20263160 | 15 | 7/10 | 1/10 |
| 20263161 | 5 | 2/10 | 2/10 |
| 20263161 | 10 | 4/10 | 4/10 |
| 20263161 | 15 | 4/10 | 2/10 |

Model predictions and actual encodings are matched to the same executed
action chain at the same time. Different arms have different physical action
chains and share roots, so pair counts are not independent experimental
replicates. Qpos distance is evaluator-only, not native success. Counts may
be sensitive to very small margins; no significance claim is made.
As a post-hoc numerical sensitivity check only, excluding pairs where either
of the two compared differences has magnitude at most 1e-4 changes just one
count in this table: root 20263160, step 15, open-vs-observed becomes 6
reversals. This absolute threshold is in each metric's own units, is not a
measurement-noise estimate, and was not fixed as an experimental criterion.

At steps 10 and 15, using the preceding real RGB/action context lowered
latent MSE to the subsequent actual encoding in 18 of 20 arm/time cases;
at step 5 open and refreshed predictions use the same starting context.
This is a prediction diagnostic for the same executed actions, not a
feedback-policy intervention or evidence of better control. The previously
recorded model selector evaluated imagined CEM suffixes, whereas this replay
uses each arm's physically executed, real-RGB-replanned suffix; its final
costs must not be substituted for selector-time costs.

The run charged three open and three refreshed predictor samples per arm;
encoding and wall-time costs are not fully certified. It used no new roots,
and no P1 gate was changed.

For the fixed selected-versus-refined-1 contrast at step 15, signed deltas
are selected minus refined 1 (negative favors selected):

| Root | Open predicted latent cost | Actual RGB latent cost | Physical qpos distance |
|---|---:|---:|---:|
| 20263160 | -0.0000437 | +0.0126808 | +0.0128447 |
| 20263161 | -0.0012963 | -0.0055100 | +0.0218562 |

Thus the first root exhibits prediction-versus-observation disagreement,
although its predicted margin is small; on the second, predicted and observed
latent favor selected while qpos distance favors refined 1. These contrasts
use each arm's executed feedback-replanned suffix and do not isolate the
cause of the initial selector decision or establish model-independent benefit.

Artifact: `le-wm-main/results/reacher_pc_feasible_prediction_replay_20261102_v1/report.json`
(SHA256 `9892d65b8c9a636914de88f83141d48fa22482ad2a97e627407240964142f73f`).
The thin five-arm wrapper SHA256 is
`bcf6f4c75f5e4c0ac408c2b007ab8bea317abbd5797f353ff8f0d0f746c049e1`;
the reused replay implementation SHA256 is
`f317179542dc6f590cdd22e522a1765c6f8086663494766010e1ac0d9d708b9c`.
