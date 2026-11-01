# Goal-only orientation contrast: raster input core, not a repaired metric

This module only generates images/arrays for a future frozen-embedding diagnostic.
No neural/physics execution, checkpoint download, readout/adapter training, Q
construction, action optimization or outcome-dependent parameter choice occurs.
Only NEW files/output; all previous sources/reports remain sealed.

## Read-only source finding

`p3p4_goal_patch_metric_20261027_v1.catalog` removes a complete isolated agent
sprite, selects14x14 cells intersecting the canonical object-color mask, and
whitens individual cells. `compile_metric` projects cell-ablation CLS responses
off a finite agent-nuisance span and takes an SVD span. It is a goal-only
appearance-response construction, NOT a test of angular discrimination or an
angle-calibrated metric. Cell deletion can change appearance without representing
an object pose. This NEW core neither modifies nor recompiles that Q.

Earlier `pusht_goal_object_tangent_metric.variants` rotates only exact-color
object-mask pixels and erases only those pixels. Outlines/antialias pixels remain
at the original orientation. Its pure-color raster is not a complete rotated
sprite; this implementation does not reuse that transform or rename its Q.

## Fixed input contract, before any future CUDA embeddings

Four pre-existing independent-goal PNGs are byte/pixel-pinned in source. No
physical reports, raw state/pose labels, action outcomes, model weights, latent
arrays or planner states are loaded. Root indices only identify source files,
not independent object orientations. The four images have the same object raster
and differ in agent placement; dedup counts are recorded, not four-root evidence.

Fixed RGB ROI: half-open `[48,48,176,176]`,128x128, on224x224 images. It was
declared after inspecting goal RGB only, not the four-root outcomes. Require
canonical object color `[119,136,153]` entirely within ROI. All nonwhite ROI
pixels after agent removal must form one8-connected component, with white
perimeter and full-angle support clearance. Preserve ALL component pixels,
including outline and antialiasing; do not rotate a pure-color mask.

Agent canonical color `[65,105,225]` locates a bbox with fixed two-pixel margin.
Require its complete perimeter white and no canonical object intersection;
erase the FULL sprite bbox to white. No hidden fill, inpainting, learned masks,
fallback box, foreground-component downselection or clipping. This succeeds on
these isolated renderer-specific RGBs, not on arbitrary scenes. Root1's agent
is inside the broad ROI; rotating the whole uncleaned ROI would rotate a nuisance
agent too. Original and clean reference images are both retained to expose this
change of scene condition. The orientation test is on an agent-removed scene,
not a claim of equivalence to the original goal scene.

Rotation pivot is the RGB ink-mass centroid of the entire isolated component
(weights=sum of channel darkness relative to white). Add0.5 to pixel-index
coordinates for PIL's edge-coordinate center. This uses only goal appearance;
it is NOT the simulator rigid-body COM or an inferred hidden pose. Require a
conservative support-radius bound (pixel-square margin plus2pixels) strictly
within ROI for ANY angle. Refuse the entire catalog if any check fails.

Full-circle uniform12-angle table is `[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150]`
degrees, symmetric modulo360;180 is its own opposite representative. No small-angle
selection, outcome-ranked subset or adaptive phase. Positive is PIL displayed
counterclockwise. Each angle has BOTH NEAREST and BILINEAR outputs, independently
retained. All pixels outside ROI equal the clean image. Zero rotation must be
byte-identical to clean for both samplers. Original+clean+24outputs=26per goal,
104total images in fixed root/angle/sampler order, including duplicate pixels.

## Faithfulness and nuisance limits

Goal-only raster orientation contrasts ARE executable here under the checked
white-background/isolation contract. Renderer-faithful physical rotations ARE
NOT established. Unknown geometry, occlusion, shadows or nonwhite ROI background
cannot be recovered from one raster. A foreign component causes refusal; a
foreign artifact connected to the object is not distinguishable with this color
contract and is outside its validity domain. No code claims generic segmentation.

Rotations resample edges, can change ink mass/centroid, and can introduce jagged
or blurred contours. Original→clean removes the agent. Those are genuine
confounds, not cosmetic details: embedding differences alone cannot establish
orientation selectivity. Nearest/bilinear pairs expose sampler dependence but
do NOT isolate it perfectly from orientation. Roundtrip(θ,−θ) RGB MSE is measured
for every sample without saving/encoding additional roundtrip images; it is an
artifact diagnostic, not a physical fidelity certificate. Record per image ink
mass ratio, foreground support count, centroid drift, changed pixels and
roundtrip MSE, plus per-angle paired-sampler RGB MSE. No fitted tolerances or
post-encoding sample deletion. A future meaningful failure test must keep these
nuisance checks visible and interpret orientation distinguishability separately
from task relevance and closed-loop benefit.

## Small executable preparation, no neural runner

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_goal_orientation_contrast_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_goal_orientation_contrast_20261027_v1 prepare \
  --output results/p1_goal_orientation_contrast_20261027_v1_prepared01
.venv/bin/python -B -m experiments.spectral_reachability.p1_goal_orientation_contrast_20261027_v1 verify \
  --output results/p1_goal_orientation_contrast_20261027_v1_prepared01 --prepare-sha SHA_FROM_PREPARE
```

Output contains104small PNGs, manifest and prepare receipt. The entire output
must be LESS THAN1,000,000bytes before any directory is created; otherwise refuse,
never reduce angles/roots. Maintain200MiB free reserve. Source/test/doc and old
generator hashes are bound in the manifest; input PNG bytes/pixels are externally
pinned; verify reconstructs all pixels, PNGs, order, artifact stats and metadata.
NumPy/Pillow versions are recorded. Output is exclusive, never overwrite.

Actual work: RGB/PIL/NumPy only,0neural/physics/AD/SVD/training calls. If main
later encodes EVERY image, cap104 batch1 frozen native encoders +104 existing
projectors,0predict/actionencoder/AD/physics; no such executable is included or
launched here. Duplicate pixels are explicitly retained and would be charged
unless main freezes a DIFFERENT, source-mapped dedup protocol before launching.
No budget equivalence with planning or past Q generation is claimed.

Pure array/image tests distinguish implementation integrity from scientific
efficacy. They cover actual-goal isolation, all angles/samplers, reference identity,
local pixel invariance, refusal cases, source/input/output drift and <1MB output.
No tests establish embedding orientation sensitivity: that needs main-reviewed
future CUDA encoding and a separately frozen numerical analysis, not a fitted
readout or immediately altered planner Q. Physical equivalence, new metric
validity, task transfer and P1/P3 admission all remain unestablished.

## Actual image-only checks, before encoding

19 tests passed on2026-10-27. The four clean object patches are byte-identical:
1unique patch; the104retained full RGB images have27unique pixel hashes, NOT104
independent poses or4independent orientation scenes. Complete ROI nonwhite support
is1429pixels while the canonical-color subset is183pixels. Source patch SHA256:
`fc17eae7d588d4d1c374d21364486328e3a02dea8a5e472b0f9d0fb7931f1f01`.

Measured over the entire fixed angle/sampler catalog (not selection thresholds):
ink mass ratio0.990047231–1.012415128; maximum ink-centroid displacement0.559114700
pixels; roundtrip ROI RGB MSE0–20.021606445 (uint8 intensity-squared units,
not normalized pixels); maximum nearest/bilinear ROI RGB MSE28.960876465.
These are real image-array measurements, NOT neural or physical results.
Whole catalog serialization is approximately305kB, well below1MB; the prepare
command reports the exact byte count after source/doc binding. No neural result
or repaired-Q score exists in this delivery.
