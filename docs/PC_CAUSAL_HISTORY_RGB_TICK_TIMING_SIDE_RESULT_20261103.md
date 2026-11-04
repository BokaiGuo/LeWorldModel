# Causal-history RGB tick timing: side diagnostic

The independently instrumented legacy executor reproduces the **current**
unmodified `NativeRGBBridge` byte-for-byte for all 12 images (four roots ×
reset/step1/step2). It does **not** reproduce the 12 saved images from the
earlier causal-history run byte-for-byte. The green-marker centroid difference
between the current bridge and saved images ranges from 0.011 to 0.275 pixels.
The rendering discrepancy is not explained; these are new replay measurements,
not direct measurements of the earlier execution.

Within this replay, reset runs `forward` before rendering, so particle geom
position is synchronized with reset qpos. At every subsequent tick, the
post-`sim.step()` geom position equals the **previous tick's qpos** plus
static offsets, maximum coordinate residual 4.45e-16 across 40 ticks. Current
qpos has already advanced by one 0.01 s integration tick; its maximum
coordinate difference from cached geom position was 0.01542 world units.
Each action spans five ticks/0.05 s. Because the bridge renders after tick 5
without another `forward`, the post-step image is rendered from cached geom
coordinates at the pre-final-tick state, not current boundary qpos. Step 1's
image therefore corresponds to tick 4 geometry, and step 2's to tick 9.
This follows from recorded MuJoCo geom coordinates, not an assertion that
pixel centroid is an exact projection under occlusion or rasterization.

Consequently reset→step1 spans geometry times 0→0.04 s, whereas
step1→step2 spans 0.04→0.09 s; both RGB differences are divided by 0.05 s
in the existing estimator. The first interval has a systematic timing
asymmetry; the second covers 0.05 s but is one tick behind qpos boundaries.
Acceleration, contact, occlusion, pixel segmentation and perspective can add
further errors. Instantaneous endpoint qvel and interval-average velocity are
different quantities even with perfectly aligned images.

| Root | Interval | RGB velocity (x,y) | qpos interval average (x,y) | End qvel (x,y) |
|---:|---|---|---|---|
| 8343 | reset→1 | (0.352, 0.161) | (0.525, 0.337) | (0.523, 0.335) |
| 8343 | 1→2 | (0.643, 0.531) | (0.519, 0.333) | (0.517, 0.331) |
| 8345 | reset→1 | (-0.625, -0.079) | (-1.534, -0.069) | (-1.527, -0.069) |
| 8345 | 1→2 | (-0.066, -0.134) | (0.178, -0.068) | (0.178, -0.068) |
| 8349 | reset→1 | (-0.845, 0.522) | (-1.094, 0.383) | (-1.089, 0.382) |
| 8349 | 1→2 | (-0.949, 0.245) | (-1.081, 0.379) | (-1.076, 0.377) |
| 8351 | reset→1 | (0.094, 0.033) | (0.055, 0.248) | (0.055, 0.247) |
| 8351 | 1→2 | (0.029, 0.220) | (0.054, 0.245) | (0.054, 0.244) |

For 8345, the x-direction even differs between the second-interval RGB
estimate and qpos-average velocity; the tick records show a velocity reversal
at the start of that interval. This discrepancy is not attributable to the
one-tick cache alone: the RGB position estimate also differs from the
rendered geom-based position (about 0.046–0.050 world units in x at steps
1–2). No corrected estimator or causal-performance claim is established.

The JSON contains all centroids, positions, tick states and per-interval
values; the NPZ contains the 12 new frames and raw state arrays. The
separate bridge check records both frame comparisons and hashes. Source,
scene, scripts and protocol hashes are bound in those outputs. Commands,
from `le-wm-main` (requires the existing legacy environment and X display):

```sh
env CUDA_VISIBLE_DEVICES='' DISPLAY=:0 PYTHONPATH=. .venv/bin/python experiments/pc_causal_history_rgb_tick_timing_20261103_side.py
env CUDA_VISIBLE_DEVICES='' DISPLAY=:0 PYTHONPATH=. .venv/bin/python experiments/pc_causal_history_rgb_tick_bridgecheck_20261103_side.py
```

Both refuse to overwrite existing results. No learned model or optimizer was
loaded; CUDA was hidden. The legacy renderer nevertheless reported using GPU 0
as a graphics device while the first process ran; this is not a CPU-only
rendering claim. No GPU model was occupied, and the worker exited. No main
code, gate or planning input was changed. Four exposed roots, one action
pair, one graphics environment and one replay cannot establish generality or
certify any historical execution's hidden state.
