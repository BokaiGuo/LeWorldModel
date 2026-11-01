# Fresh source-native PushT factual pilot

Implementation and non-physics preparation only. Main alone installs/reviews
the existing runtime overlay and explicitly runs this collector. Old runtime
and archived episode1 seals remain unchanged. No neural imports/forwards,
training, optimization, control comparison or P5 admission.

## Fixed execution

Only seed2026102701, two independent official PushTEnv instances a/b, identical
explicit constructor configuration from the runtime proposal. Constructor
calls seed(None); record its RNG state, then explicitly seed2026102701 BEFORE
the one reset. Authoritative root is post-reset: native `_set_state` advances
physics once at .01s. Each env then executes15 explicit float64 raw arguments
`u=[0,0]`, NOT None and NOT normalizedzero. In predictor units z=(u-mean)/std;
native step scales u by100 and holds the current positional target for10
substeps. Zero means PD damping/contact evolution, not paused physics.

Collect both complete trajectories before comparing; no favorable-root
selection, replacement, retry or mismatch filtering. Unexpected execution,
I/O, dependency, observation or counter failure produces failed report with
all partial evidence retained. Incomplete intent is indeterminate, not a
license to resume/replay into the same output directory.

## Raw evidence and exact identity

Each of32 native RGB observations is saved losslessly as uint8224x224x3 `.npy`,
with native float32 state7/proprio4, raw transition return and full-public-state
JSON. No extra render/_get_obs calls reconstruct observations. Every command
has a durable file binding the raw argument, actual agent position, absolute
PD target and previous public-state SHA. Constructor/pre-reset RNG states are
also retained. RNG arrays carry dtype/shape; JSON preserves float64 precision
and explicitly encodes infinities (e.g. kinematic mass), rejecting NaN/object
data. No pickle, MP4 identity evidence or unsafe decoder.

Full-public-state captures agent/block/static body position, velocity, angle,
angular velocity, force, torque, CoG, mass/moment/type/sleeping; every native
shape's semantic geometry, radius, filters/friction/elasticity/surface velocity,
sensor/collision type/color; space gravity/damping/solver/contact settings,
RNG, goal/config/contact counters and coverage history. Semantic shape sorting
removes address/set iteration order, retains duplicates. It does **not** capture
private Chipmunk solver caches or serialize an arbitrarily restorable Space.
No pointer IDs or nondeterministic constructor seed are comparison targets;
fixed-seed post-reset RNG/state is. This does not hide state mismatches.

All16 paired frame indices must match state/proprio/RGB/full-public-state AND
transition file SHA exactly. No tolerance or threshold tuning. Both sides'
entire assets remain even on mismatch. Fresh simulator repeatability is the
only scientific scope: no archived episode reconstruction, encoder equivalence,
prediction accuracy, action fidelity or prospective control benefit claim.

## Journal and cost boundary

Hash-chained append-only journal entries are flushed/fsynced BEFORE each
construct/seed/reset/step/close; directory entries and all saved evidence are
fsynced too. A native control-step intent predeclares its10substeps; it does
not claim atomicity against interruption inside that native call. Returned
events and assets-durable events distinguish completed calls from persisted
observations. Never automatically retry an incomplete intent.

`sys.setprofile` observes original Python function entries and every Space.step
dt without wrapping or monkeypatching native methods. Expected totals:
2env constructs,4seed calls (2constructor+2explicit),2resets,30control calls,
302Space.step(.01),32native renders,32native _get_obs,2closes; temporary native
goal bodies and other observed calls are also counted. Per env1.51s includes
reset. Profile hooking is an explicit instrumentation boundary, not an encoder
or simulator shim. No neural calls; no additional physical branches.

## Prepare/tests/main command

From `/home/gbk/code/LeWorldModel/le-wm-main`, original environment, no overlay:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_factual_pilot_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1 prepare
```

After main's review, use the manifest SHA delivered with this new version:

```sh
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1 run --confirm-physics --manifest-sha256 MANIFEST_SHA256
```

Default new output directory:
`results/p5_jepa_factual_pilot_20261027_v1/pilot_main_v1/`.
Collector reruns the strict runtime import-only probe and saves its receipt,
checks all source/config seals before/after, requires exact target import
origins and inherited constraints. It imports official source explicitly after
the probe (two class imports, neither constructs physics); no source patch.
Output directory must be exclusive; existing output is never overwritten.
Success/failure writes `report.json` plus hash-bound `journal.jsonl` and both
raw asset trees. Preflight failure constructs no environment. Tests use only
plain property/constant-array fixtures, never an actual physics implementation;
passing tests/prepare are not a factual physics run.
