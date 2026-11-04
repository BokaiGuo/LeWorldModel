# RGB task-region spectral candidate development

The fixed development protocol generated action directions from the frozen
model's full spatial Jacobian and from that Jacobian restricted to tokens
near the RGB-detected current and goal markers. A seeded random direction
was the third family. All three used the same task-region nonlinear rollout
score, five prefix candidates and eight feedback-tail candidates. Each
selected chain was executed in the native RGB environment with real step-5
feedback; final position was evaluated by the separate offline physics
backend.

| Development root | Task-region spectral | Full-spatial spectral | Random |
| --- | ---: | ---: | ---: |
| 1884 | 1.32083 | 1.75838 | 0.01172 |
| 1885 | 0.83022 | 0.29650 | 1.69078 |
| 1886 | 1.41366 | 1.41194 | 0.75414 |
| 1887 | 1.69825 | 1.73004 | 0.96072 |

Smaller is closer to the archive frame-10 position. Task-region spectral
is closer than full-spatial spectral on two of four roots and closer than
random on one of four. The four roots were already used for development;
this is not an independent test or evidence of a reliable improvement.
The task mask is a hand-designed PointMaze RGB color/geometry heuristic,
not a learned critic or a general semantic task representation. Missing,
oversized, broadly fragmented or edge-adjacent marker masks are rejected;
the bounding-box check does not certify connected components or all possible
segmentation errors.

The two spectral families use the same two JVPs and separate SVDs; random
does not require that structure calculation. All families use equal
candidate counts, but this is not an equal-total-compute comparison. There
was no strong CEM/GRASP baseline or reconfiguration intervention. Position
distance is not native success, and the newly generated chains' position
labels were not cross-backend checked against live hidden states. Model
proprio is fixed to raw zero, the full archive supplies normalization
statistics, training membership is unknown and the camera was calibrated on
an exposed root.

All prefix banks, scores, task masks, images, full action Jacobian columns,
singular values, endpoint labels and several helper hashes were saved.
Three targeted tests passed across marker fail-closed checks, mask/Jacobian
and SVD recomputation, saved RGB hashes, selection indices and distance
arithmetic. They do not independently rerun the model or certify every
possible segmentation condition. The unchanged original experiments remain
intact. This is not a P1 pass, complete task-weighted D2P, or proof against
the underlying mechanism.

- Protocol: `docs/PC_POINTMAZE_RGB_TASK_SPECTRAL_DEV_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a1_pointmaze_rgb_task_spectral_dev_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_rgb_task_spectral_dev_20261102_DEV01.json`
- Result SHA-256:
  `e87a066b6a49ac10753e5a656347e5bf3f8d08b1263c981f02b065278b8797c5`
- Tests: `le-wm-main/tests/test_a1_pointmaze_rgb_task_spectral_dev_20261102.py`
