# P1 prefix-null same-action model error: handoff

This is an exploratory saved-data evaluation over all 16 physical rows (eight
selected-prefix / zero-prefix pairs), without outcome selection, training,
optimization, physics, or new roots. Several rows share action/task identities;
they are retained and labelled, not independent replications. The prior
`p1_backtrack_rgb_feedback_cuda_20261030.py` is bound to a different physical
report and must not be treated as the result of this evaluation.

The fixed physical report SHA is
`751344ac69ac078322592707dcd89eb534ee777e63128b3a8cd6298251953551`.
The existing v2 saved consumer checks journal, physical action bytes, state,
native termination and images before preparation. Each row is joined by
task, model root, blind source, physical root and condition to the historical
model input; warmup raw bytes must normalize exactly to the two model history
tokens. Checkpoint PNG file and decoded RGB hashes are verified again.

For each row, encode actual frames 0/5/10/15 on CUDA. Open-loop predictions
propagate the three original latent-history elements over the same three
five-action packets. Refresh uses actual encoded frame history
`[old_history[1], RGB0, RGB5]` to predict 10, and `[RGB0, RGB5, RGB10]`
to predict 15, with identical outgoing action histories as the open branch.
Step 5 has no refreshed prediction. Latent MSE and L2 compare each prediction
against the same-time actual RGB encoding; this is not pixel-space error or
proof of physical predictive accuracy outside these trajectories. Original
history and actual RGB0 may differ; neither is silently replaced in the open
branch. No goal encoding is required. Costs: 64 encoder/projector and 80
predictor/action-encoder calls, without training. Source and model hashes,
row-level predictions, target latents, normalized tokens and action contexts
are saved. No P1 admission or independent confirmation follows.

## Main-thread host handoff

Review code and tests, confirm the source and model seal, sufficient host GPU
memory and explicit exclusive GPU coordination. The following commands are
**instructions only**, not executed by this work. Work from
`/home/gbk/code/LeWorldModel/le-wm-main` in an authorized host terminal:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -s -m experiments.spectral_reachability.codex_p1_null_same_action_error_cuda_20261030_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_null_same_action_error_cuda_20261030_v1_prepare_MAIN01.json
```

Record and independently review the printed manifest SHA, then provide it as
`REVIEWED_PREPARE_SHA` in the following command. Use the same source files
and environment; the program rebuilds and compares the complete manifest.
Never use a previous or partial output as a new attempt.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -s -m experiments.spectral_reachability.codex_p1_null_same_action_error_cuda_20261030_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_null_same_action_error_cuda_20261030_v1_prepare_MAIN01.json --prepare-sha "$REVIEWED_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_null_same_action_error_cuda_20261030_v1_run_MAIN01.json
```

CUDA initialization and an actual allocation are required; CUDA-disabled or
unavailable environments fail without CPU neural fallback. The script has no
independent proof of host identity or exclusive GPU ownership; those are
operator gates. The run creates its output exclusively before model loading.
A failed process may leave an empty or partial file, which must be retained and
reviewed; it is not success and cannot be silently overwritten. This module
does not implement a separate saved-result consumer or cross-entrypoint
global lock. Run only after main-thread review of these limitations. Numerical
and source tests do not establish CUDA runtime compatibility.
