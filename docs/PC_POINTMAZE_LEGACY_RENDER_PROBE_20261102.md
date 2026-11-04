# PointMaze legacy renderer compatibility probe

This follow-up renders the same four previously exposed archive roots
(1880–1883), frames 0/5/10, using the official visual XML and local
`mujoco-py` 2.1.2.14. The backend follows the environment's render sequence:
create a render context, set azimuth 90 and elevation -90, reset the
recorded state, move the hidden target offscreen, and call `sim.render`.
Only the image/state matching question is tested; no neural planner runs.

| Fixed output | Mean absolute pixel error, 0–255 | Fraction of differing pixel locations |
| --- | ---: | ---: |
| Legacy output | 17.5839–17.7900 | 34.04%–34.47% |
| Legacy output vertically flipped | 18.9255–19.2278 | 35.85%–36.30% |
| Previous MuJoCo 3.1.6 free-camera probe | 38.9933–39.2082 | 70.26%–70.42% |

Both legacy orientations are reported; neither is adopted as a deployed
preprocessing rule. The actual legacy camera was azimuth 90, elevation -90,
distance 9.73485, look-at `(3.0, 2.5, 0.0)`. The previous modern probe used
distance 14.60227 and look-at approximately `(0.8674, 0.8674, -0.1)`.
Thus the two probes differ in camera setup as well as renderer/backend, and
their improvement cannot be attributed solely to a MuJoCo version change.

The legacy rendering is closer in this fixed pixel measure, but still differs
substantially. Archive camera settings, pixel transformations, texture
generation and training preprocessing are not yet authenticated. No visual
equivalence tolerance was set in advance. These outputs cannot yet replace
real archive-compatible feedback or support a closed-loop admission claim.
The saved `versions.runtime` is the string returned by
`mujoco_py.get_version()`, not an independent runtime-version attestation;
the result separately hashes the loaded runtime library path.

The initial execution exposed a parser mistake: `get_version()` returned a
string, which the parent tried to convert to an integer. It failed before
writing a report; the parser was corrected and the full fixed set rerun.
This was a diagnostic-runner error, not an experiment outcome.

- Runner: `le-wm-main/experiments/a0_pointmaze_legacy_render_probe_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_legacy_render_probe_20261102_DEV01.json`
- SHA-256: `6de7face4573089fcac9d8bc817f98a6819a7294409de9ef9387ee124ca96925`
- Three focused tests passed across the native and legacy probes. They check
  source/row binding, output ranges and XML sites, not pixel equivalence.

The next engineering step is to identify the archive's authoritative data
generation and preprocessing configuration and verify it on held-out frames.
If it cannot be recovered, use a newly generated native-render environment
with its own fixed model-input contract rather than claim archive equivalence.
Neither this diagnostic nor the earlier physical oracle establishes P1.
