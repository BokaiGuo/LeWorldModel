# Post-outcome four-trajectory prefix intervention

This is an exploratory hypothesis generated after observing RUN01, not an
independent confirmation or a P1 admission. Use all four saved trajectories
in their fixed order; do not choose roots, arms, or suffixes by outcomes.
For each trajectory, its executed steps 1–5 are the selected prefix and its
executed steps 6–15 are the saved suffix. Cross the selected, all-zero, and
previously certified greedy prefixes with that same saved suffix and a
ten-step zero suffix. The fixed denominator is 4 × 3 × 2 = 24 branches.

The greedy prefix is the fixed root-slot-0/off action from the earlier
222-candidate predicted-stage-cost certification, not a newly optimized
prefix. Its original task ID differs from RUN01's task ID. The prepare requires
matching physical root hash, state, goal, root RGB and dynamics, while retaining
both task IDs and their separate provenance. Selected-prefix branch replay
also requires the saved RUN01 step-5 physical state, snapshot and RGB;
zero/greedy replay requires the corresponding saved step-5 reference.

Each branch resets and verifies the actual historical warmup/root. Actions
are exact float32 raw commands; no model inference or new candidate selection.
Native `info.n_contacts` is saved for each returned public control step as the
environment's aggregated count, not as contact geometry or a causal proof.
An original guard failure is censored only if the actual returned state,
branch, action, time and preserved RGB establish finite out-of-canvas state.
Other errors fail closed. Early stop and censored endpoints remain missing;
all 24 requested branches remain in the roster.

This protocol is observationally motivated; neither its fixed order nor
source hashes retroactively establish preregistration or freshness. The
durable physical-operation meter counts its exposed boundaries, not total
compute, costs of historical candidate generation, or all internal operations.
This is not budget fairness or evidence of a strong baseline. Disk and wall
limits are cooperative; failed I/O may prevent a final report. Source
checks and replay seals do not prove every transitive dependency or causal
mechanism.

No GPU or physics has been run for this new protocol. The prepare command is:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_prefix_intervention_20261030 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_prefix_intervention_20261030_prepare_CODEX03.json
```

After the new source, inputs, tests and printed prepare SHA have been
independently reviewed, an authorized human can substitute that SHA below.
The output path must be unused. Verify this job will not interfere with
other active tasks.

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_prefix_intervention_20261030 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_prefix_intervention_20261030_prepare_CODEX03.json --sha256 PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_prefix_intervention_20261030_run_MAIN01 --authorization MAIN_AUTHORIZES_POSTHOC_FOUR_TRAJECTORY_PREFIX_PHYSICS
```

The current implementation leaves the run `recorded_not_independently_consumed`
until a separate saved consumer exists. Do not treat a zero exit as a validated
physical finding.
