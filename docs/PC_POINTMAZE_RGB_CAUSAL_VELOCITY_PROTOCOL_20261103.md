# Causal RGB velocity and model feedback: development protocol

Use the four previously exposed roots and both sealed first-packet actions
from `a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json`
(SHA-256 `ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade`).
All three arms replay the same selected first packet in the native RGB
environment, requesting an RGB observation after each of five high-level
actions. The final two observations at cp4 and cp5 are therefore causal and
available equally to all arms; check cp5 hashes against the sealed source.

For the point marker, use the fixed analytic RGB-to-position projection
already recorded in `PC_POINTMAZE_RGB_GEOMETRY_STATE_RESULT_20261103.md`.
Use the scene's documented MuJoCo timestep 0.01 and frame skip 5 to compute
`(position_cp5-position_cp4)/0.05`. This is a finite-difference average,
not certified instantaneous velocity: raw rendered geometry may lag the
integrator, pixels are quantized, and collisions can occur. Compare:

1. fixed raw-zero proprio;
2. RGB cp5 position plus zero velocity;
3. RGB cp5 position plus the causal cp4-to-cp5 velocity estimate.

At cp5 all use the same frozen model, RGB goal, feedback task mask, tail CEM
noise, three generations of 16 candidates, horizon and time budget category.
Each selected tail is then executed for five actual steps. Only after all
selections are complete, replay action sequences using the audited
position-only physical backend, to evaluate goal-frame-10 distance and
estimated cp5 state error. The privileged backend state is never consumed
by any model scoring arm. Record each RGB frame/hash, action, estimated
state, predictions, physical outcome and costs. Compare the raw-zero arm's
tail, model score and final RGB hash against its sealed source.

The fixed first action is not the one that a new RGB-state planner would
necessarily select. These are exposed development roots, scene-dependent
camera metadata and a normalizer with unverified training provenance;
position distance is not native success. Equal prediction counts and frame
access do not certify equal runtime. The study is neither an independent
closed-loop benchmark nor evidence of reconfiguration or P1 passage.
