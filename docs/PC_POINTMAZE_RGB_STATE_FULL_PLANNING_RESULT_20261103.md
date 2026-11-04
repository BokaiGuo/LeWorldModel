# Initial and feedback RGB-position planning: development result

The prewritten `PC_POINTMAZE_RGB_STATE_FULL_PLANNING_PROTOCOL_20261103.md`
ran on host CUDA on four previously exposed development roots. For each root,
four arms independently generated a first plan, executed five native
high-level actions, encoded the real cp5 RGB, replanned, and executed another
five actions. The zero-state arms reproduced both saved plans and cp5/cp10
RGB hashes from the previous run. The new RGB arms use fixed analytic
position and assume zero velocity at both decision points.

Each number below is the evaluation-only distance from the position replay
backend to the frame-10 goal, lower is better. It is not a native success
rate, an independent evaluation, or an intervention on P1's
reconfiguration mechanism.

| Root | Zero / random CEM | Zero / spectral CEM | RGB / random CEM | RGB / spectral CEM |
| --- | ---: | ---: | ---: | ---: |
| 1884 | 1.0266 | 1.6790 | 0.3938 | 0.4123 |
| 1885 | 0.9938 | 2.4003 | 0.1692 | 0.0431 |
| 1886 | 0.8204 | 0.9543 | 0.4018 | 0.4108 |
| 1887 | 2.0130 | 1.6068 | 0.6820 | 0.7148 |

At each root and for both search choices, RGB position with zero velocity
has lower replay distance than the corresponding zero-state arm. With RGB
position, spectral initialization is better than random CEM on one of four
roots and worse on three. These are four exposed roots, not sixteen
independent outcomes; no inference or P1 passage is justified. The result
supports investigating input-interface mismatch, not concluding that the
spectral mechanism succeeds or fails generally.

Each arm performs 48 initial and 48 tail model-prediction samples. Spectral
arms additionally perform two JVPs and one SVD (measured structure time is
about 0.042--0.071 seconds here); therefore these arms are **not equal
total compute**. Encoding, CEM and structure timings are recorded
separately, not treated as a complete end-to-end cost certification.
At cp5 each arm receives its own executed RGB; comparisons do not hold the
prefix fixed and cannot isolate causal effects of a common prefix.

The renderer may lag integration, the analytic projection relies on fixed
scene/camera metadata, and the zero-velocity assumption is not a
measurement. Reset state is used to initialize the environment and the
position backend is used only for evaluation; neither is passed to the
selector. The goal, archive and checkpoint training provenance remain
development limitations. The feedback encoder uses the implemented
single-image context, not a certified full action-history interface.
Further independent-root, native-success, resource-matched and action-history
tests are required.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json`:
  SHA-256 `4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99`
- NPZ `le-wm-main/results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.npz`:
  SHA-256 `bd3a50d329b14551b98ea9cffae3da1a366b727e3ff34b4873b70ad80cdb8181`
- `.venv/bin/pytest -q tests/test_a1_pointmaze_rgb_state_full_planning_20261103.py`
  from `le-wm-main`: **2 passed**. The test checks saved inputs, CEM
  selection replay, cost counts, identities and hashes, not new independent
  physical or GPU execution.
