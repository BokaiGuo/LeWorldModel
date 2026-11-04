# PointMaze native-XML render compatibility probe

This development-only probe addresses a prerequisite for a model-in-the-loop
remaining-horizon versus extended-horizon comparison: can the online
environment provide images compatible with the recorded PointMaze archive?
The four roots 1880–1883 and frames 0, 5, 10 are previously exposed; they are
not an independent outcome panel.

The probe extracts the released `point_maze` and `parse_maze` functions and
their original `dynamic_mjc` dependency to construct the visual XML. It
sets each recorded state, moves the target site offscreen as the source
environment does without a displayed target, and renders with MuJoCo 3.1.6
using the free-camera defaults plus the source `prepare_for_render`
azimuth 90 and elevation -90. It does not fit camera parameters to the
archive frames. This is **not** the original legacy renderer: camera defaults,
MuJoCo version, rendering details and preprocessing remain unverified.

Across the 12 frames, mean absolute error was **38.9933–39.2082 per
channel on the 0–255 scale**; approximately **70.26–70.42%** of pixel
locations differed in at least one channel. These numbers establish that
this particular reconstruction is not pixel-identical; they do not identify
which camera, renderer, texture, image orientation or preprocessing
difference caused the mismatch. No image similarity tolerance was specified
in advance, so there is no quantitative equivalence-admission decision.
The current reconstruction must not be called archive-matched or silently
substituted into an RGB-feedback confirmation.

The existing position-only backend's `_legacy_pointmaze_xml` omits the
original visual assets and particle/target sites; it is appropriate only
within its documented position-label contract, not as an archive-camera
rendering source. Conversely, visual equality alone would not certify
dynamics, action timing, or hidden simulator state.

Also, in `select_fixed_tree_action` of the existing
`run_sra_p4_pointmaze_rgb_closed_loop_cuda.py`, the branch labelled
`spectral` selects the minimum two-step latent cost from a fixed five-action
bank. That function does not compute an action Jacobian, SVD, or
dynamics-generated spectral direction. Its result therefore cannot by
itself establish the proposed spectral-program mechanism. That runner also
uses the `DatasetMatchedRenderer`, an approximation built from calibration
images and state-to-pixel mapping; it is not native environment camera RGB.
These observations describe the implementation, not a new outcome experiment.

Artifacts:

- Runner: `le-wm-main/experiments/a0_pointmaze_native_render_probe_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_native_render_probe_20261102_DEV01.json`
- Result SHA-256:
  `a75c6aad6c0cb4a02bef0ede544b0b139cd218bd7f7419858a1c57b83acc4982`
- Two targeted tests passed, checking the official XML visual sites and the
  fixed 12-row saved-result contract. They do not validate render equivalence.

Next use the actual legacy renderer and original camera/preprocessing
configuration, if available, and compare held-out recorded frames before
deploying its observation in a feedback experiment. Until then, any
dataset-matched synthetic image experiment must be explicitly labelled as
such and cannot establish native RGB-only closed-loop performance.
