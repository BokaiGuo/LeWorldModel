# Paired spectral shadow run: offline physical replay

Replay all four sealed 17-action development trajectories, without selecting
or changing actions, from their saved setup states. The replay has two jobs:

1. Use the same MuJoCo 2.1.2 shadow bridge and XML to replay every saved
   action and verify every returned RGB frame against the saved 18 frames.
2. In a separate evaluator process, use that same MuJoCo 2.1.2 model,
   reset, pre-step forward, velocity clipping and five integrator steps to
   obtain per-step physical position and velocity, without rendering into the
   physical simulator. Also run the established MuJoCo 2.2 position-label
   backend and report both endpoints and their differences.

Report physical Euclidean position distance to the fixed PointMaze goal at
steps 0, 12 and 17, plus RGB-proxy score and per-step states. Goal and state
are evaluator-only, never returned to or consumed by the planner. Verify
trajectory hashes and actions and freeze all input/code hashes. This is
deterministic replay and backend cross-check, not a direct log of the original
simulator's privileged state; RGB replay does not itself prove latent physical
state identity. Do not call position distance native success. Both roots were
exposed during development and resources were not matched in total.
