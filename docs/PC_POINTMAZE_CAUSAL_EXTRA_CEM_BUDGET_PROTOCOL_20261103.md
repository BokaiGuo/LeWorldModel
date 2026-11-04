# Causal-RGB planner with extra random-CEM search budget

Before outcomes, fix archive roots 1888–1895, start frame 0 and goal RGB
frame 10, one frozen PointMaze checkpoint, scene and camera, and deterministic
per-root innovations seeded `20261103 + root`. These roots appear in prior
project material and are **development data**, not certified fresh or
training-disjoint. Do not select roots by outcomes, replace failures, or claim
independent confirmation.

On each root execute two planning arms with identical initial RGB-derived
position and zero velocity, initial RGB goal mask, real per-step observation,
initial two-packet and tail one-packet horizon, and causal cp4→cp5 RGB
velocity. The spectral arm executes 3×16 initial CEM samples with the
existing four first-generation JVP/SVD injections and two JVPs plus one SVD.
The random arm executes 4×16 initial samples, sharing the first three
innovation batches and using a fourth fixed batch. Both use the same 3×16
tail innovations and optimize their own real-RGB feedback state. Record
separate encoding, structure, CEM, prediction sample/transition, and physical
execution costs; 16 extra random samples are a **chosen stress test**, not a
demonstrated equal conversion for JVP/SVD or equal wall-time budget.

Use RGB, actions, frozen model and static scene metadata only for decisions.
Load privileged target state for evaluation only after all actions are chosen.
Execute ten steps without early stopping; record offline strict terminal
distance `<0.5`, not claimed native online success. Replay actions using the
legacy renderer and position backend and require matching terminal RGB and
positions. Persist every trace, frame and actual count, including failures.
Random ordering is fixed and shared; no outcome-based tuning.

Compare terminal distance and endpoint-threshold results by root. This test
does not establish non-greedy reconfiguration, a fair equal-total-budget
win, independent generalization, or P1 admission. Full checkpoint training
provenance and historical root reuse remain uncertified.
