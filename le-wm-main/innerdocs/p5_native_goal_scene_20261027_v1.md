# Independent native goal-scene generator (not executed by this agent)

New source/test/doc only. No old sealed edits or downloads. Preparation reads
source/runtime metadata only and does not import native physics or neural code.
The main-only `run` entry is separately authorized, reviewed and invoked by main;
tests use only static source inspection and arithmetic/metadata fakes, never a
real native environment. No generator physics or neural call was run in delivery.

## Prior scene definition, not a selected future image

Object goal is the unchanged native [256,256,pi/4], actual source T polygons,
default COG and `legacy=False`. Agent nuisance is fixed [64,448] with zero velocity,
a canonical lower-left scene location, not derived from any factual root/frame
or selected after looking at predictions. Scene seed2026102799 is independent
asset-generation RNG control, not a new evaluation root/cohort. Only one scene,
one attempt; no image screening, alternate agent placements, retries or goal
replacement. Nothing reads the nonzero factual trajectory, frame15, v11 costs,
features, predictions, or new branch outcomes.

Use official native constructor with reset_to_state=[64,448,256,256,pi/4,0,0],
with_velocity/target=True, relative=True, action_scale100, render224 and unchanged
default shape/color/damping/COG. Native reset calls `_set_state`, which advances
Space once by .01s. Therefore main goal generation IS physics work: exactly
one constructor, two seed calls (constructor seed(None), then fixed seed), one
reset/space step/render/get_obs/close and ZERO env.step controls. It is NOT a
physics-free image synthesis or a prospective physical control episode.

After saving the authoritative native RGB/state/proprio/full-public-state,
validate the agent position, object body origin/angle and zero velocities.
Construct the official `_get_goal_pose_body` and compare every actual block
Poly vertex through live `block.local_to_world` versus official goal-body
`local_to_world`, fixed atol1e-10/rtol0. Retain local/world vertices and both COGs.
No T-shape guessing, center-distance substitution, image compositing, camera
patch, extra physics advance or extra render. Any mismatch fails closed and
retains the actual returned assets; do not adjust reset pose to make it pass.
Geometry alignment is not an observed coverage/success measurement: no control
step ran, native coverage history is empty. Do not synthesize coverage=1.

## Durable output and bridge interface

Source/test/doc/native-source seal and the immutable prior plan are saved BEFORE
main constructs a native scene. A hash-chained/fsynced journal records every
operation intent/return, durable assets and terminal status. Explicit expected
prepare-file SHA plus main authorization and exact physics overlay are required.
No torch-loaded process or CPU neural fallback is permitted.
The no-torch sys.modules guard remains before preparation/runtime loading and
is checked again after native dependency import, before any scene construction.
An independent-process test verifies module import plus prepare leaves no torch
modules loaded; the native import probe itself is NOT executed by that test.
Existing sealed
pilot helpers are reused by explicit arguments, never global patches. The
native runtime import probe is main-only and separately retained. Constructor
RNG is recorded before fixed seed overrides it. Close/final-integrity failures
retain the original capture failure rather than replacing it.

On success, emit native frame_00 assets, geometry.json, generation.json,
goal_spec.json, report.json and journal. `generation.json` and `goal_spec.json`
match the already frozen `p5_prospective_control_bridge_20261027_v1` schema.
They bind the independently declared pose, nuisance, actual RGB file, generation
source/prepare hashes, geometry file hash and actual native-call census. Only a
successful generation emits a bridge goal spec, committed LAST after durable
terminal journal/report and generation receipt; the generation receipt binds
the execution report and journal hashes too. Interrupted publication cannot
authorize compilation through an absent spec. Prepare emits NO image or fake
generation receipt. Failed native counts/source/pose/geometry do not authorize
bridge compilation or become a negative scientific result.

The bridge still needs NEW reviewed CUDA compiler and physical executor. This
generator does not provide either. Main passes the independently generated goal
spec into a fresh bridge prepare, compiles using only RGB/proprio0/5/10 plus two
history packets and the separate goal, freezes the selected full five-command
packet and all three branch slots, then independently resets/seeds/replays
prefix10 for compiled/nominal/factual_reference. No factual frame15 fitting is
relabelled as control. Agent nuisance in goalRGB is not an agent success target;
object XY/rotation/coverage and agent motion must remain separately reported.
No C2/P5 mechanism or generalization claim follows from this asset/bridge.

## Commands

Project root `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_native_goal_scene_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p5_native_goal_scene_20261027_v1 prepare --output results/p5_native_goal_scene_20261027_v1_prepare_v1
```

This is the only actual prepare run in this delivery. Main, AFTER source review,
may compute the protocol.json SHA and run the following SEPARATE physics command:

```sh
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_native_goal_scene_20261027_v1 run --prepared results/p5_native_goal_scene_20261027_v1_prepare_v1/protocol.json --expected-sha256 ACTUAL_PREPARE_FILE_SHA256 --output results/p5_native_goal_scene_20261027_v1_scene_main_v1 --main-authorized
```

ACTUAL_PREPARE_FILE_SHA256 is a required independently retained digest, not a
fixture hash. No physics command has been run by this agent; availability of
source and passing fixtures does not prove native scene construction succeeds.
Reuse/overwrite of any prior output is forbidden. Freeze source after tests;
any change to the prior scene or implementation needs a new sealed version.
