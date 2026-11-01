# Four-arm fixed executed-action model versus RGB

The existing known-root common-optimizer run has four completed trajectories,
five saved RGB checkpoints per arm, and each arm's four selected/executed
action packets. It records candidate scores, but not the selected one-packet
prediction vectors required for a direct prediction-versus-RGB comparison;
the terminal checkpoint also has no subsequent online boundary encoding.

The new entry rebuilds all 16 original node contexts from saved feedback and
verifies the fixed report and prepare, physical journal chain, warmup and
control action bytes, receipt/task identity, RGB files and encoded latents.
Its `inspect` command runs no neural model or physics. On a separately
coordinated HOST CUDA run, it predicts one packet from each reconstructed
context using **that arm's own executed packet**, and re-encodes the matching
next RGB frame with the same frozen model. This is within-trajectory
same-action/same-time alignment, not a common identical intervention across
the four arms.

For every arm and boundary the output retains both 192-dimensional vectors,
their residual, MSE and L2; relative to the fixed latent goal it also records
actual-goal MSE, predicted-goal MSE and the signed cross term. The identity
`predicted_goal_mse = actual_goal_mse + residual_mse + cross_term` is algebra,
not a causal decomposition. Four terminal frames are re-encoded as part of
the fixed 16 target encodings. Existing encodings at steps 5/10/15 are not
silently substituted for these new encodings.

The output is exploratory, post-hoc, single-known-root and not independent
confirmation. Neither saved physical authenticity beyond the checked records
nor runtime hardware identity across processes is independently proven.
There is no equal-compute or P1 admission claim. Cost events record attempted
and returned outer calls with synchronization and elapsed time; internals,
FLOPs, GPU time, full loading and upstream experiment costs remain unknown.
Checks use the earlier 1800-second, 3 GiB allocated, 4 GiB reserved CUDA
limits, sampled between operations rather than enforcing interruption inside
a long operation.

CPU-only check (no model or simulator):

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.codex_four_arm_same_action_prediction_20261030_v1 inspect
```

After separate GPU coordination, runtime/environment review and choosing a
new exclusive output name, the host operator may run:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.codex_four_arm_same_action_prediction_20261030_v1 run \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_same_action_prediction_20261030_v1_HOST01 \
  --authorization MAIN_AUTHORIZES_FOUR_ARM_FIXED_ACTION_PREDICTION_CUDA_ONLY
```

The host command was not run during this work. Failed runs keep their own
partial journal and exclusive output directory; do not overwrite them.
