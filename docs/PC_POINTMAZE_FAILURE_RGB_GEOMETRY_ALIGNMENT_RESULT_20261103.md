# PointMaze: real RGB geometry versus task outcome

## Scope and result

The preregistered [diagnostic](PC_POINTMAZE_FAILURE_RGB_GEOMETRY_ALIGNMENT_PROTOCOL_20261103.md)
projects all 624 sealed terminal images and their goal images using the
existing RGB marker detector and static camera; it does not change the
controller, rerun physics, fit a metric, or select a subset of images.
All terminal and goal detections succeeded. The 12 candidate banks come
from three previously exposed, outcome-selected development roots.

The most discriminating case is root 1893, random-extra tail, 48 candidates:

| Pairwise comparison | Reversed / comparable |
| --- | ---: |
| RGB geometry distance versus physical position distance | 2 / 1128 |
| Real RGB latent goal score versus RGB geometry distance | 767 / 1128 |
| Real RGB latent goal score versus physical distance, prior sealed diagnostic | 769 / 1128 |

The RGB goal projection error against the evaluation-only physical goal is
0.0156; mean and maximum terminal position errors are 0.0489 and 0.0759.
The lowest RGB geometric distance corresponds to a physical distance of
0.46994, whereas the prior real-RGB latent-score argmin had physical
distance 1.249. These are **future-image oracles**, not choices available
to the planner before acting.

Across the 12 banks, RGB/physical ranking reversals range from 2 to 118;
each bank has 1124–2016 comparable pairs. Goal projection errors are
0.0156–0.0590, and bank-wise mean terminal projection errors are
0.0286–0.0489. These quantities share candidates and trajectories and are
not independent statistical samples.

## Interpretation and limits

For this exposed bank the position-based RGB metric agrees substantially
better with physical position than the frozen model's *real-image* latent
goal score. A large target-coordinate error is therefore not needed to
explain that particular latent/physical ranking disagreement. This does not
establish why the latent score differs: renderer timing, image appearance,
encoder representation, task metric and projection error remain possible
contributors. It also does not show that RGB geometry applied to imagined
futures is deployable: the world model predicts latents, not the true future
images used here.

The detector, camera and offset were developed on exposed data; marker
occlusion, pixel quantization, and render/integration timing remain limits.
The observed bank is a retrospective diagnostic, not independent
confirmation, a fair planning comparison, a successful control result, or
evidence of P1's non-greedy mechanism. Any planning intervention must use
only currently observed RGB and action history, charge its computation, and
be evaluated on newly fixed trajectories without using simulator state or
future images to select actions.

## Provenance and verification

- Report: `le-wm-main/results/a0_pointmaze_failure_rgb_geometry_alignment_20261103_DEV01.json`,
  SHA-256 `bb4f05528cf2f97cac8d0f9a55dd935370f823b853c0ed662c74f1d2c0c6708d`.
- Arrays: same basename `.npz`, SHA-256
  `a6c9f14bfbf0d2b6050dd2e799661d0493cc15e5933492af74bfbbbec1ebb6d7`.
- The report binds the protocol, script and input hashes; the prior latent
  decomposition is in
  [real RGB metric result](PC_POINTMAZE_FAILURE_REAL_RGB_METRIC_RESULT_20261103.md).
- Saved data and all 624 image projections, rankings, goal errors and oracle
  selections were recomputed by
  `tests/test_a0_pointmaze_failure_rgb_geometry_alignment_20261103.py`.
  Together with the prior saved-RGB test: **2 passed**. This verification
  does not rerun the world model or physics.
