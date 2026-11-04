# PointMaze model-scored old-tail retention development experiment

Fix the four exposed roots and both previously selected initial plans from
`a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json` SHA-256
`ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade`.
Replay the saved model-selected first packet in the native RGB environment,
check reset and cp5 image bindings, then encode actual cp5 RGB and score
one-packet tails using the same frozen model, cp5 RGB task mask and goal.

At each root and initial arm compare cold CEM versus old-tail retention.
Both use the saved tail noise, three generations, 16 candidates per
generation, four elites, identical bounds and variance floor. Warm CEM
replaces one first-generation candidate with the prior plan's remaining
packet; that proposal is scored inside the same batch and counted in 48
samples, not silently added. Execute each selected tail from a fresh reset
and matching first packet in the RGB environment, save images and full
traces. Use the position evaluator only after selection. Score and action
trace reproduction against the old cold arm is checked; frozen weights,
source arrays, actions and reset hashes are checked.

Record shared initial candidate costs, feedback encodings, tail model
prediction counts, wall time, and spectral JVP/SVD separately. Matching
prediction counts are not a total-compute certificate. The existing model
uses raw-zero proprio and full-archive normalization, whose checkpoint
training provenance remains unknown. Position distance is not native
success; the position backend and native RGB bridge have different
implementations. These are exposed development roots; old-tail retention was
motivated by earlier outcomes. Do not claim independent confirmation,
reconfiguration mechanism, or P1.
