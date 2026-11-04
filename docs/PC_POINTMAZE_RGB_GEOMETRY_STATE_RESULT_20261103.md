# Static-camera RGB position and unknown velocity

An analytic projection was built from the native legacy camera's actual
distance `9.7348469`, 45-degree field of view, top-down orientation and
static scene's particle-body offset `[1.2, 1.2, 0]`. It segments the green
marker in the current RGB image, converts its pixel centroid to position and
sets velocity to zero because no pre-root observation exists. It does not
fit state-labelled samples or use simulator state to construct this
`rgb_position_zero_velocity` input. The `(3,3)` camera look-at had already
been selected on exposed root 1880, and the scene geometry is privileged
environment metadata; portability to a different camera or scene is unknown.

On the four previously exposed roots, RGB position error versus archived
initial position ranged from **0.00397 to 0.01181** world units. On the
eight fixed candidate banks, model scoring with RGB position and zero velocity
selected a previously executed candidate with smaller position-replay
distance than the raw-zero-input choice in seven groups; root 1885 random CEM
was worse (`0.34878` versus `0.27883`). The RGB-position choice matched the
privileged-position/zero-velocity choice in seven of eight groups.

This is a fixed-bank, post-outcome development comparison: the candidates
and their physical outcomes already existed. It is not a new trajectory,
closed-loop comparison, native success rate, independent confirmation, or
P1. The privileged full-state comparator remains substantially better in
several groups; unknown velocity still matters, and an RGB/action-history
estimator must be tested at later timesteps with causal observations.
The same full-archive normalizer and fixed checkpoint interface retain their
previous source limitations. The RGB marker segmentation rejects absent or
oversized detections but does not yet authenticate all occlusion conditions.

The first metadata attempt failed to parse an extra legacy-library shutdown
message; the worker parser was fixed to accept exactly one JSON line and the
successful report was sealed afterward. Three tests passed across camera
projection, absent marker and invalid camera rejection, saved input binding,
candidate-selection arithmetic and the prior privileged diagnostic.
These tests do not independently rerun the frozen neural model.

- Protocol: `docs/PC_POINTMAZE_RGB_GEOMETRY_STATE_PROTOCOL_20261103.md`
- Camera metadata runner:
  `le-wm-main/experiments/a0_pointmaze_camera_metadata_20261103.py`
- RGB estimator and comparison:
  `le-wm-main/experiments/a0_pointmaze_rgb_geometry_state_20261103.py`
- Result SHA-256:
  `1f93bc226fb5b7cdb2bfa2b20b0faaa6b62d86d0871d04f0874e138eb33a9d30`
- Arrays SHA-256:
  `d28b49ceebeb627ede5acf52a58384f387d825ce4b1a6a6d47e5e420d12b410e`
