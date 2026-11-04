# PointMaze development camera-offset diagnosis

The prior legacy-render mismatch was localized to the static maze as well as
the moving point. On root 1880 at frames 0/5/10, the union of pixels that
changed across either image sequence occupied 0.879% of the frame, yet mean
absolute error outside that union was 17.405/255. The montage shows the
rendered maze vertically displaced relative to the stored maze. This motion
mask is only a localization aid, not a semantic segmentation.

An exploratory five-point camera look-at grid was then evaluated on the
already exposed root 1880 frame 0. Keeping the legacy backend, distance,
azimuth and elevation fixed:

| Look-at x,y | Mean absolute pixel error |
| --- | ---: |
| 3.0, 2.5 (legacy default) | 17.7235 |
| 2.5, 2.5 | 32.5720 |
| 3.5, 2.5 | 32.5159 |
| 3.0, 2.0 | 32.0924 |
| 3.0, 3.0 | **0.0000** |

The fixed `(3,3)` look-at selected from this development frame was then
checked on three other previously exposed development roots (1881–1883),
at frames 0, 5 and 10. All three frame-0 renders were byte-identical to their
archive frames. The six later frames had 98–129 differing pixel locations,
with mean absolute error 0.0091–0.0484/255. Thus the camera offset explains
the major initial visual mismatch for this setup, but later frames are not
byte-identical. These are **not fresh scientific confirmation roots**, and
the selected camera remains a development-derived parameter rather than an
authenticated archive-generation configuration.

The remaining differences might involve saved-state precision, frame/state
timing, rendering, or another interface detail; this diagnostic does not
identify which. Before reusing the camera in a planning experiment, compare
actual action replay with stored image/state timing and test the full
observation-to-model interface. Renderer agreement alone does not certify
the physics, training data split, RGB-only input, or independent P1.

Artifacts and SHA-256:

- Difference diagnostic: `le-wm-main/results/a0_pointmaze_visual_difference_20261102_DEV01.json`
  (`2e643faff4f87e95ae4e0be088bc4a7fcc34bdec25483b7e23608e7acf04a439`).
- Camera grid: `le-wm-main/results/a0_pointmaze_camera_offset_probe_20261102_DEV01.json`
  (`1c0fff1cc2e2ea3feee1e0cc1e798c2baa2e969230d9563fc0b5dd737bde0e06`).
- Fixed-camera validation: `le-wm-main/results/a0_pointmaze_camera_validation_20261102_DEV01.json`
  (`89a87203999b382a764b15da399a515199db0ba2ce6a6dcac661ffc20787d2d6`).
- Montage: `le-wm-main/results/a0_pointmaze_visual_difference_20261102_DEV01.png`
  (archive, legacy, absolute difference in columns; frames in rows).
- Five targeted probe tests passed. They verify fixed identities and saved
  source bindings, not the original camera provenance or planning efficacy.
