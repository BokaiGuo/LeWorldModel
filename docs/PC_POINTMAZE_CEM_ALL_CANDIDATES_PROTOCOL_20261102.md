# PointMaze common-CEM fixed-candidate diagnostic

Development-only protocol, fixed before the new physical executions. Source:
`le-wm-main/results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json`,
SHA-256 `ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade`.

For each source root 1884–1887 and both random and spectral CEM arms, execute
all three generations of 16 saved initial two-packet proposals from the
same initial state, in saved order (384 trajectories, including duplicates).
Each packet is its saved float32 action repeated for five physical steps;
there is no cp5 replanning in this diagnostic. Verify source/array hashes,
model identity, reset images and matching selected-prefix images. Use the
source initial RGB-derived mask and goal to calculate actual cp10 RGB feature
cost on the frozen host-CUDA model. Independently replay identical actions
with the position evaluator and report goal-position distance as an
evaluation-only physical outcome.

Within each root/arm, compare saved predicted cost, observed RGB feature cost
and position distance on the same time and actions. Report pairwise reversals
and ties, the model-selected candidate, the actual-feature and physical
candidate oracles, and their physical-distance regrets. Record the complete
candidate actions, images, scores, endpoints and provenance. Oracles use real
future outcomes and are not deployable planners. Source CEM best initial
plan is not the original feedback-replanned chain; do not compare the new
cp10 image against the old feedback execution unless its tail matches.

The four roots are already exposed development data; proposals share roots,
generations and potentially identical actions. Pair counts are descriptive,
not independent observations, significance evidence, P1 confirmation, native
success, or a strong-baseline comparison. The source checkpoint's proprio and
normalizer provenance remain uncertified. Preserve all negative results.
