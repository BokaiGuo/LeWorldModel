# Saved feedback object motion versus fixed goal: geometry-only audit

New module/tests/innerdoc/output only. Uses experiment-code's real saved-data,
provenance, deterministic numerical test and report pattern. No applicable
AGENTS.md was found in the workspace ancestor/source/test/doc paths. Central
progress and all previous sealed sources/reports/documents remain read-only.
No model, neural forward, gradient, training, simulator, Space, reset, physics,
render, encode, download or outcome-selected threshold. This is an arithmetic
consumer; figures or a new efficacy experiment would not add evidence here.

## Inputs and audit binding

Input is the completed six-branch contact-recovery report
`results/p4_feedback_contact_recovery_20261027_v1_main_v1/report.json`.
Verify its source/archive/args seal, recover its original physical report path,
and rerun ONLY the old `recovery.audit` pure saved-data arithmetic. Require exact
agreement of six rows, geometry, paired roles, pose contract and compiler hash.
All old and fresh read-input hashes are checked again and tracked to final exit.
This retains compiler/feedback contexts, fixed Q/latent target, physical root goal,
raw actions, original prefix replay, pixel gate, all state/RGB/journal receipts,
and saved encoding arithmetic without new neural or physical execution.

All three roles x imagined/actual are retained, including aliases and adverse
results. Require completed snapshots0..10 in every branch:66 receipts,60 actual
primitive intervals,30 suffix intervals. Never replace step10 with a shortened
endpoint. The six original physical goal states must be exactly identical.

## Coordinate conventions: origin is not COG

Freeze old geometry.contract before reading outcomes. Its reviewed official
source SHA, real two-polygon T, radius15, and collision_slop0.1 are unchanged.
Use the official unattached Pymunk Body/Poly transforms: mean of the two official
Poly centroids for COG, set snapshot angle then snapshot block.position, and
local_to_world/world_to_local. No Space or dynamics. COG local is (0,45) in this
environment; it is not local origin and is not replaced by an area-weighted
centroid. Snapshot block.position is body local_to_world(0,0). Report both origin
and world COG; do NOT subtract COG again from the saved origin. State[-2:] is
agent velocity, not block velocity, and is not used as measured object motion.

Axes are native saved x/y, without display y-flip or clockwise/anticlockwise
physical claims. Rotate vectors, not positions, into the previous body axes.
Agent local-coordinate change across rotating frames is not agent displacement:
report actual world displacement plus that vector expressed in previous body
axes, and separate endpoint local agent coordinates. Angles use atan2(sin,cos)
principal differences; mark half-turn goal ambiguity. Multiple unseen revolutions
and substep trajectories cannot be recovered from these snapshots.

## Fixed-goal alignment, exact progress and rigid shape

For each interval, plus net prefix0->5 and suffix5->10, report actual agent,
object origin and COG displacement, angular change and shortest desired angular
change to the same fixed goal. For origin and COG separately, report dot,
signed projection, perpendicular component and cosine versus the before-state
goal vector, AND exact distance reduction before-minus-after. A positive
projection can overshoot and worsen exact error: keep both. Undefined cosine
for exactly zero displacement or zero goal vector is null, not zero or success.
No movement cutoff is tuned from observed outcomes.

Track the same eight external T boundary vertices in local order at each pose
and at the goal. Report corresponding-vertex displacements and goal RMS before/
after, not a closest-shape rematching that erases rotation. This RMS is an
explicit shape diagnostic, not official success or a new goal threshold.
Verify exact rigid decomposition for every vertex about BOTH the body origin
and the COG: delta_vertex = delta_reference + (R_delta-I)*(vertex_before-reference).
Record each reconstruction residual. Numerical tolerance1e-9 concerns arithmetic
only, not physics/contact efficacy. The original pose remains position distance
+20*absolute wrapped angle; split its change into position and angle contributions.

## Pusher displacement at geometric anchors, not measured contact forces

For every interval, retain both endpoint near labels (old clearance<=0.1), not
only favorable/contact-selected steps. At both before and after geometry, find
the pusher-center nearest point on the external T boundary, record edge ID and
edge fraction, then carry that SAME local material point through before, after
and goal poses. Do not reselect a new nearest point and call its difference
material motion. Stable first-index edge ties are explicit; corner normals may
be nonunique. Boundary winding is verified before deriving a geometric outward
edge normal and tangent; neither is a measured force direction.

Project finite pusher displacement, material-point displacement, and their
difference onto the endpoint inward/tangent directions; record pusher alignment
with actual object origin/COG motion, that material point's motion, and fixed
goal directions. These are observed finite-displacement relationships, not
instantaneous velocity, pressure, slip, force, lever-arm torque or push causality.
Intervals with either endpoint near are separately counted/aggregated with
defined cosine counts; they are not declared continuously contacting intervals.
No interpolation or causal attribution is performed.

## Reporting and scientific boundary

Six per-step/full-net outputs and summaries retain exact distance/angle/vertex
progress, path lengths and original pose change. P1/P4 admission remains false,
force_causality and actual_contact_during_interval remain unknown regardless
of alignment. One reused root, no uncertainty/equivalence/held-out efficacy claim.
The audit can distinguish translation away from goal and rotational progress,
or co-moving pusher/object snapshots, without concluding why forces acted.

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_motion_alignment_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_motion_alignment_20261027_v1 --recovery-report results/p4_feedback_contact_recovery_20261027_v1_main_v1/report.json --output results/p4_feedback_motion_alignment_20261027_v1_main_v1
```

Output must be a fresh tagged directory. Seal new module/test/innerdoc and explicit
old arithmetic/geometry sources before reading outcomes; exclusively fsync
report.json/report.md and verify the seal afterward. No overwrite or fallback.
Exit0 means saved audit completed only; exit2 retains a new integrity_fail output.
Synthetic tests cover known translation/rotation, overshoot, frame invariance,
COG/origin distinction, material anchors/corner ties, missing data, source/receipt
tampering, and nonneural imports. A real saved-panel regression separately checks
all six branches and all60 intervals. No fixture is presented as new physics.
