# History-CEM trajectories: legacy endpoint replay and wrapper evaluation

The prewritten `PC_POINTMAZE_HISTORY_CEM_NATIVE_ENDPOINT_PROTOCOL_20261103.md`
replayed all 16 sealed ten-step action sequences using the same legacy
MuJoCo RGB backend as the executor and the separate position backend.
All 16 cp5 and cp10 RGB hashes reproduced, and all position-backend states
reproduced their saved endpoint exactly. The two replay backends' final
positions and velocities were identical on all 16 trajectories (maximum
component-group L2 differences `0.0`). The original RGB-only bridge did not
expose hidden state, so this is deterministic **replay**, not direct
recording of original simulator state or general backend equivalence.

The checked-in `PointMazeWrapper.eval_state` defines success as strict
position distance `< 0.5`; the underlying sparse reward uses `<= 0.5`.
Against the archive frame-10 goal state, both single-frame and history
planners reach the wrapper threshold on 6/8 fixed-prefix trajectories.
All four arms for roots 1884, 1885 and 1886 pass, and all four arms for
1887 fail. Thus the eight paired binary outcomes tie; four roots, not
sixteen independent tasks, underlie these results. The exact distances are
reported in `PC_POINTMAZE_RGB_HISTORY_CEM_FEEDBACK_RESULT_20261103.md`.

The wrapper also reports full-state distance, which includes velocity; it is
saved separately and is not the success criterion. The original executor
ran all ten actions and did not stop when the threshold was crossed;
endpoint success is not an early-stopping trajectory. These are exposed
development roots, the prefix was fixed from prior experiments, and history
and single-frame costs differ. No independent generalization, equal-budget
advantage, non-greedy mechanism, or P1 passage follows.

- Receipt:
  `le-wm-main/results/a1_pointmaze_history_cem_native_endpoint_20261103_DEV01.json`,
  SHA-256 `7e924dd570c3315cd758d64cf3ab88a9e344d7768579f548b5d5299aee8e2b90`.
- `.venv/bin/pytest -q tests/test_a1_pointmaze_history_cem_native_endpoint_20261103.py`:
  **2 passed**, including the strict threshold edge case and saved arithmetic.
  This test reads the receipt; it does not independently rerun the simulator.
