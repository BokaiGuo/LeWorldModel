# RGB-position actual-history feedback CEM: physical development result

The prewritten `PC_POINTMAZE_RGB_HISTORY_CEM_FEEDBACK_PROTOCOL_20261103.md`
ran on host CUDA. On four exposed roots, each of two sealed RGB-position
prefixes was replayed with two tail planners, yielding 16 native RGB
trajectories. Both branches in each pair execute the same five-step prefix
and match the sealed cp5 RGB. Single-frame tails, scores and cp10 RGB hashes
all reproduce the source; history tails were newly optimized from the real
cp0/cp5 frames and the actual prefix packet, then physically executed.

| Root | Fixed prefix | Single-frame distance | History distance |
| --- | --- | ---: | ---: |
| 1884 | random CEM | 0.39375 | 0.39576 |
| 1884 | spectral CEM | 0.41229 | 0.41431 |
| 1885 | random CEM | 0.16924 | 0.14553 |
| 1885 | spectral CEM | 0.04313 | 0.01254 |
| 1886 | random CEM | 0.40178 | 0.35045 |
| 1886 | spectral CEM | 0.41083 | 0.41119 |
| 1887 | random CEM | 0.68200 | 0.71608 |
| 1887 | spectral CEM | 0.71480 | 0.68142 |

Distances are evaluation-only position-backend replay distances to the
frame-10 goal, lower is better, not native success. History yields lower
distance in four pairs and higher distance in four. All eight history arms
select a different tail. Pairs share four roots and prior development
decisions, so these counts do not provide independent statistical evidence.
This is a real execution of newly optimized tails, but it does not test
the P1 reconfiguration mechanism or establish a planning advantage.

Each tail optimizer scored 48 model samples in three batches. History CEM
took approximately 0.224--0.226 s per pair in this run versus approximately
0.079 s for seven warmed single-frame calls (the first single call took
0.172 s). History also used an extra initial frame encode. These measured
times are not a controlled equal-total-budget comparison, and do not
include all upstream prefix generation or environment costs. The model
weights were unchanged; RGB position assumes zero velocity, and the
checkpoint training provenance and history contract remain unverified.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_rgb_history_cem_feedback_20261103_DEV01.json`:
  SHA-256 `a5cf40bad026652440dff1c123a489ae9f80deeb5102c8d6a8b79f29ad3d3932`
- NPZ `le-wm-main/results/a1_pointmaze_rgb_history_cem_feedback_20261103_DEV01.npz`:
  SHA-256 `8e295f7a8df4c753598ea32d5077a9c9f6c7cfe065c2ee36881dd306b34309e9`
- `.venv/bin/pytest -q tests/test_a1_pointmaze_rgb_history_cem_feedback_20261103.py`:
  **1 passed**, checking hashes, same-prefix bindings, CEM replay and
  evaluation arithmetic, not independently rerunning the model or physics.
