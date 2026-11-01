# P3/P4 task-metric repair hypothesis: goal-only patch interventions

Implementation-thread predeclared under authorized implementation, following
the known-root P3/P4 metric/pose discordance. Not a root-cause finding, not a
new independent cohort, not a prospective efficacy claim about already viewed
outcomes. No learned readout/adapter, training, test-state scoring or source edits.

## Evidence and non-duplication

Installed env.py reset first sets agent/object to goal_state and renders goal;
the render includes agent. The four byte-bound original goal PNGs contain
39/44/47/43 pure RoyalBlue agent pixels and183 pure canonical object pixels
each. Their isolated agent sprite boxes, including margin/outline, are
[20,166,37,182], [66,102,83,119], [120,186,137,203], [30,17,47,34].
All counts/masks are RGB evidence, not physical pose labels. Complete sprite
removal is possible here only under the checked white-background isolation
condition. Pure-blue-only erasure leaves outlines/anti-aliasing and is not used.

Existing files already implement finite agent goal-span Q, six object-shift/
rotation tangent Q, current+goal scene nuisance Q, and full-sprite goal catalogs.
These are not renamed as a new repair. Existing artifacts show construction
integrity, not genuine task-metric correctness or a certified physical cause.
Old Q removes a finite global nuisance span, not all agent sensitivity; removing
more rank can also discard real object information. Metric disagreement alone
cannot distinguish prediction error, task mismatch, local linearization failure,
velocity nuisance or other explanations. Official success includes agent, while
object pose deliberately does not. Do not substitute one for the other.

The released checkpoint is tiny ViT,224x224,patch14:256 patch tokens and CLS,
width192. Installed LeWM.encode takes last_hidden_state[:,0] and applies the
frozen projector. Predictor/pred_proj output only one projected192-dimensional
state per boundary. Patch tokens are available for encoded images, but their
future values are NOT predicted. Averaging goal object patch tokens and comparing
them to predicted CLS is illegal. So is applying an untrained CLS projector
independently to patch tokens and claiming equivalence. Learned bridges/readouts
or simulator-state costs would require different authority and are excluded.

## New minimal native-space construction

Freeze the binary canonical-object mask from independent original goal RGB only.
Select every14x14 cell with any object pixel, row-major. No mask from a current,
predicted or test-physical image, no mask expansion chosen from outcome labels.
The goal mask and cells are hashed before encoding and never updated downstream.
Require isolated complete agent sprite with a two-pixel white perimeter and no
object-color overlap. Remove full sprite to white; preserve object pixels exactly.
Refuse clipping, unknown backgrounds, missing colors or corner placement overlap.
No fallback placement or reconstructed hidden object. This is renderer-specific
RGB editing, not a certified distribution-preserving physical intervention.

Goal encoder catalog: original, source-agent-removed goal, four complete sprites
at fixed centers(28,28),(196,28),(28,196),(196,196), then each selected object
cell individually whitened in the cleaned goal. All images otherwise unchanged.
Original native CLS must match original goal0; fixed model/projector never changes.

N includes original5 agent-goal differences, original-minus-clean and four
corner-minus-clean vectors. One SVD yields U_N. O contains each cell-ablation
CLS minus clean CLS. One SVD of (I-U_N U_N^T)O yields U_T. New Q=U_T U_T^T.
Use fixed relative cutoff1e-6, with machine-roundoff floor
max(matrix_shape)*float64_eps*Frobenius_norm(unprojected_source_directions).
This prevents roundoff from a fully nuisance-contained object response being
declared a new task direction; it is not fitted to outcomes. Use an
orthoprojector, no outcome scaling or
fitted weights. Zero surviving rank strictly refuses. New score remains
mean((Q@(predicted_native_CLS-original_goal0))²). Q annihilates original-clean
direction up to recorded numerical precision, so no hidden target replacement.

This is a frozen-model intervention-span projector, not learned pose decoding.
Spatial patch membership defines which RGB experiments generate native CLS
directions; it does NOT assert spatial independence after ViT attention.
Patch deletion changes appearance, not a pose coordinate. The resulting span
might miss angle, emphasize appearance or fail outside goal neighbourhood.
Reduced responses/rank are not proof of task selectivity or physical benefit.

## Frozen comparison and executable path

First test is score-only: old Q and new Q on the exact same per-root union of
sealed v4 uniform/dynamic candidate prefixes. Deduplicate by(horizon,pool index),
requiring identical full actions/context witnesses across arms. Stable selection
is cost then H then pool index. No new samples, learned readout, adaptive mask,
latent endpoint edits or physics outcomes. Different Q scales are not compared
as effect sizes; compare selected controls and independently measured physical
outcomes later. Retain actual selected H2/H3 and partial strata.

--compile-cuda is a real MAIN-only HOST CUDA goal-encoding executable, with
pre-dispatch encoder cap, durable attempts/completions and immutable per-root
metric/selection files. It forbids predicts and emits physical_ready=false.
No model/simulator is imported by --prepare, --verify or array tests.
This workstream DOES NOT execute --compile-cuda or any physics/neural code.

New selected controls are NOT yet deployable: they need independent double
neural replay of each selected path, a new source-bound exporter and new physical
consumer. Reusing the old physical consumer whitelist or labeling inherited
candidate witnesses as fresh double replay would be wrong. Main can first review
the compiled model-only artifacts, then authorize that separate execution bridge.
No root outcome is used to include/exclude cases or choose a Q variant.

For later P3/P4 compilation, compile_action_response is an executable array
bridge: apply exactly the same new Q to nominal residual, native action Jacobian,
finite-H2/H3 residual signal and bounded full-joint DLS(.002, sqrt(5H)*.75).
Its returned proposal requires nonlinear evaluation before any winner/dispatch.
Old raw Jacobians may be reused only at their original context/zero point;
P4 fresh RGB contexts require fresh native action derivatives, not old root J.
Changing only final scoring but retaining old metric's residual/compiler is
explicitly a score-only ablation, not a complete repaired P3/P4 algorithm.

## Budgets and scientific limits

Goal compile per root:6+Kgoal_cells batch1 visual-encoder forwards, no predicts,
the same6+K existing frozen CLS-projector forwards (independently counted), no
action-encoder entries, no neural AD/Jacobian/VJP, two metric SVDs. All encoder
attempts and SVD attempts/completions are durable, including failures. Four-root
K and exact total are fixed in prepared receipt, not selected from neural output.
Source exporter validation adds8 saved-array SVDs, charged separately. Source
candidate prediction/AD/context generation costs remain inherited, shared once,
not zero or equal to visual compilation. Native action-response bridge costs
two array SVDs per requested H(one residual-span SVD and one bounded-ridge SVD)
plus ridge bisection, not neural work or a new Jacobian acquisition. Its cost is
not silently added to the score-only execution where it is not called.
Selection counts2N Q evaluations and2(N-1) comparisons per root. Source checking,
PNG generation and serialization are CPU overhead, not FLOP-matched resources.

Before any future new physical outcome: freeze both metrics and all commands;
execute both at same root/context, preserve variable horizon and guards; evaluate
original official, object pose, and actual native-Q scores independently. No
post-outcome Q/mask/threshold tuning. Failure to change or improve controls is a
valid negative result, not a reason to add another learned adapter. P1/P3/P4 false.

Commands: module experiments.spectral_reachability.p3p4_goal_patch_metric_20261027_v1.
--prepare --output NEW_PREPARED. --verify --prepared PATH --prepare-sha256 SHA.
Reviewed MAIN only: --compile-cuda --prepared PATH --prepare-sha256 SHA
--output NEW_COMPILE_PATH. Never overwrite/reseal old sources or results.
