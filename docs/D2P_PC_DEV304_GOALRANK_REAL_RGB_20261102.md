# Goal-ranked candidates: real RGB and physical endpoint decomposition

This post-outcome development analysis encodes the actual terminal RGB for
all 18 previously fixed and executed goal-ranked candidates. It uses the
same frozen checkpoint and goal as the model run; CUDA accounting recorded
19 encoder and projector calls (one goal and 18 candidate frames), with no
predictor or action-encoder calls and matching before/after model hashes.
The upstream saved verifier checks candidate actions, frame identity, and
the fixed ten-step execution. This is not a new action-selection experiment.

For each feedback context, the model-selected spectral action had lower
predicted and real RGB goal-latent MSE than its anchor, but a larger final
object-angle error. Object-position error was lower for both; native reward
improved for original and worsened for coherent. Thus matching the latent
direction on these two examples does not establish pose-task improvement.

Among 72 within-context candidate pairs, model and real latent goal-distance
rankings disagreed for 19; real latent ranking disagreed with physical
position for 29, angle for 20, and negative reward for 21. Exact ties were
zero. These pairs share one known root and common prefixes; they are not
independent observations. Counts include arbitrarily small nonzero margins.
They should not be compared as independent statistical rates to a different
candidate bank, nor attributed uniquely to model error or metric mismatch.

For the two selected actions the model's linear-versus-nonlinear displacement
relative errors were approximately 0.337 and 0.491, at the actual bounded
actions. This describes model-space approximation only, not physical error
or a validated trust radius. Anchor ratios are undefined and recorded null.
No physical task success, reconfiguration mechanism, equal-budget advantage,
or independent confirmation is established.

Artifacts relative to `le-wm-main/results/`:

- Encoding contract SHA-256
  `bda962802ec9615775f9233da0d45aac65120367a8a11c82c91cd67ac32d34e2`.
- CUDA encoding report
  `p1_dev304_goalrank_real_rgb_20261102_RUN_DEV01/report.json`, SHA-256
  `c486fd1a7d44e737852c597cdcbeb31ceb3cbe575291733fd7d14e09c2d829e8`.
- Saved analysis `p1_dev304_goalrank_real_rgb_saved_20261102_DEV01.json`,
  SHA-256 `7ab6ec15af1ae82cd061ef54881b303ece23d4d0bdf31fdd185c7c89d1f66e94`.
