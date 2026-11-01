# Minimal prospective native C1 architecture bridge (prepare/freeze only)

This is a new calibration interface, not P5 mechanism/generalization admission.
Main owns a NEW reviewed CUDA compiler and independent native physical executor.
This module imports neither torch nor native physics and has no execution entry.
It does not reproduce the parallel Euclidean/array audit: no feature MSE, JVP,
Gram, ridge or derivative arithmetic is recomputed here. Those remain the actual
CUDA compiler and independent array-audit responsibilities.

## Input and scientific boundary

Fixed known seed2026102701 and original nonzero prefix10, with actual native
RGB/proprio frames0/5/10 and exactly two between packets. Same-seed roles a/b
are deterministic identity repeats, not independent roots. Preparation authenticates
the collector's source seals, hash-chained journal and all32 durable frame assets.
It verifies v11's completed unchanged-model identity, not its ranking or feature
arrays. The compiler bundle contains ONLY three RGB/proprio frames, raw history
packets and the separate goalRGB. It must not call v11's four-frame factual loader,
v8's archived-data prepare/run or v9's observed-target residual fitting.

The old factual frame15 is already observed, and both human operators know its
diagnostics. This is prospective only for NEW counterfactual physical actions,
not a blind held-out future/cohort. No outcome-dependent root, seed, metric,
candidate-bank or reference selection is permitted. Frame15/state15/proprio15,
v11 predictions and all new branch outcomes are forbidden compiler data inputs.

The independent goal is the native fixed object pose [256,256,pi/4], supplied as
a separate 224x224 uint8 native goal-scene RGB asset. Agent placement in that
scene is an explicitly disclosed nuisance, NOT an agent task success target.
No goal image is synthesized by this prepare module. Main must produce it with
a separately reviewed native goal-scene generation source and receipt, before
compilation and without deriving it from observed factual frame15. Rendering
that asset is separately accounted work, not one of the three physical branches.
The producer may not claim physical branch execution or unseen-goal provenance.

`goal_spec.json` exact schema:

```json
{
  "schema": "p5_prospective_control_bridge_20261027_v1.goal",
  "rgb_path": "/absolute/path/to/new/native_goal.npy",
  "rgb_sha256": "FILE_SHA256",
  "object_goal_pose": [256.0, 256.0, 0.7853981633974483],
  "agent_goal_role": "scene nuisance; no agent success claim",
  "origin": "independently_declared_native_goal_scene",
  "generation_receipt": "/absolute/path/to/new/generation.json",
  "generation_receipt_sha256": "FILE_SHA256"
}
```

Generation receipt: `schema=TAG+'.goal_generation'`, matching `goal_rgb_sha256`
and `object_goal_pose`, `used_factual_future=false`,
`method='separate native goal scene; no branch execution'`, and nonempty
`source_bindings={absolute_source_path:sha256}`. Source/procedure review remains
necessary: these attestations and declared data-read lists are not an OS access
sandbox or proof against a dishonest compiler. Factual-directory goal paths and
pixel-identical copies of ANY saved factual frame are rejected.

## Minimal main CUDA compiler contract

Freeze protocol/source/model/data hashes first. In the original neural environment
(no physics PYTHONPATH), use unchanged v11/v8 official native model loading ONLY;
all modules frozen eval, HOST CUDA only, no training or CPU fallback. Encode three
context images plus independently supplied goal in one encoder batch; encode only
three context proprio vectors, with no invented goal proprio. Main reuses the
sealed source-native action/proprio normalization and its dataset rounding
convention, not another normalization audit or new statistics.

At decision10, nominal is five native env arguments [0,0], not normalized zero
and not None. JVP ten native normalized coordinates around that nominal, same
two history packets and context proprio. Reuse unchanged v8 .01 damped ridge
machinery, but the residual is **independent goal features minus nominal**, never
factual frame15 features. Two candidates only: nominal and one full ridge step;
nonlinear goal-feature global MSE argmin, nominal-first exact tie. No tuning,
restarts or physical scores. Convert the candidate to actual native-u units.
The explicit bridge support is |u|<=1, not the native advertised absolute-position
Box. Any nonfinite/out-of-support candidate aborts the compile, without clipping
or substituting a favorable candidate. Exact CUDA replay of the selected packet
is required before freeze, including when nominal wins.

Expected successful completed counts: encoder1/4images, proprio1, predictor13
(nominal1 + JVP10 + ridge nonlinear1 + selected replay1), JVP columns10 and
autograd.grad20 under the existing double-backward JVP implementation. Main must
record attempted/completed phases and failures/partial costs durably. No failed
receipt authorizes physics or becomes a theory-negative result.

Call `compiler_bundle(protocol, independently_retained_digest)` to read just the
seven allowed data files (three RGB/proprio pairs + goal). `digest` is the
module's canonical JSON digest of the entire prepared protocol, including its
source seal. Compiler receipt schema consumed by `frozen_branches` is executable
in the module/tests: `schema=TAG+'.compiler'`,
`status='selected_packet_replayed'`, matching `prepared_sha256`,
`compiler_inputs_sha256`, `compiler_data_bindings` exactly equal to the allowlist,
`goal_rgb_sha256`, `compiler_source`, `model_before/model_after`,
`backend='host_cuda_only'`, `used_observed_frame15=false`, `physical_calls=0`,
`attempted/completed` matching the frozen counts, `partial_work_unknown=false`,
`selected_replay_exact=true`. Candidates in fixed nominal/ridge order each carry
`id`, `env_packet[5,2]`, `independent_goal_mse`; retain `selected_id` and
`selected_env_packet`. Nonempty `artifact_bindings` bind actual output files;
`artifact_roles` maps response_columns/goal_features/nominal_successor/
ridge_successor/selected_replay/gram/rhs/ridge_delta to those paths. The bridge
checks source/input identity, scores' argmin and packet binding, NOT array math.

## Freeze first, then three independent physical executions

`freeze` reproduces prepare, verifies source/data/goal bindings and main's compiler
receipt/artifact hashes. It writes selected_packet.json, all_branches.json and
finally freeze_receipt.json with fsync and exclusive creation. All THREE slots
compiled/nominal/factual_reference must exist durably before ANY new branch
constructor/seed/reset/step. Keep roles even if packets coincide; never replace
them to make the comparison more interesting. The factual reference is the
unchanged predeclared collector suffix10..15, not a trajectory selected by MSE.

Main's independently sealed executor must revalidate the freeze commit and all
referenced sources/hashes immediately before work. For each arm in fixed order,
construct a fresh native environment with the recorded constructor arguments,
explicitly seed2026102701 BEFORE reset, execute the original ten-command prefix,
and verify all eleven saved frame0..10 RGB/state/proprio/full-public-state/
transition identities. Do NOT restore seven state coordinates or JSON public
properties as hidden Chipmunk state; do not reuse the previous arm's live env.
Then execute exactly the frozen five suffix commands, observing/saving every
step. No feedback reselection, target fitting, endpoint shortening/padding or
root replacement. Stop that arm on guard/integrity/termination, retain all slots
and distinguish statuses. Journal/fsync command intents before dispatch and
actual returns/assets afterward; uncertain partial internal work stays unknown.

Complete three-arm census: construct3, explicit+constructor seed6, reset3,
control_step45 (prefix30+suffix15), space_step453 including three reset .01s
steps, render48/get_obs48, close3. Extra goal generation/preflight work is reported
separately, never hidden in these counts. Main must implement and review executor
guards/counters; this prepare-only module does not falsely claim to enforce
runtime physics through a receipt alone.

## Agent, object and task outcomes stay separate

Existing saved public float64 traces show object motion in0..5, but exact object
pose constancy at every saved step5..10 and10..15. In the factual suffix10..15
agent displacement is about (3.379,-0.900), object displacement/rotation zero,
native coverage zero. Therefore “the whole factual trajectory only moved agent”
is false, but “the factual suffix moved agent without saved object pose progress”
is supported. This check does not infer force causality or fit motion thresholds.

Future comparison must retain object-origin XY error, wrapped rotation error and
native coverage/final-max separately from agent displacement/velocity. Native
coverage success threshold is .95; native `done` is hardcoded false, so it is not
a success flag. Mixed-scene patch distance and agent-inclusive state distance
cannot stand in for object success. No new weighting, threshold tuning or agent
target is introduced. C2 remains NOT ESTABLISHED: no non-greedy prefix, own-state
future-capability improvement and closed-loop task-benefit conjunction has been
demonstrated. Even successful single-root C1 execution would not open P1/P5 or
establish mechanism, generic action fidelity, unseen-root or architecture-wide
generalization.

## Commands and actual preparation status

From `/home/gbk/code/LeWorldModel/le-wm-main` (no physics overlay):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_prospective_control_bridge_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p5_prospective_control_bridge_20261027_v1 prepare --output results/p5_prospective_control_bridge_20261027_v1_prepare_v1
```

The no-argument-input prepare records real prerequisites/diagnostics but explicitly
blocks on missing independent goal and reviewed new compiler/executor. It is NOT
a CUDA/physics-ready claim. After main supplies those NEW files, create another
exclusive prepare with `--goal-spec /abs/spec.json --compiler-source /abs/new.py
--executor-source /abs/new_executor.py`. After main CUDA compilation, freeze with:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p5_prospective_control_bridge_20261027_v1 freeze --prepared /abs/ready_prepare/protocol.json --compiler-receipt /abs/new_cuda_receipt.json --output results/p5_prospective_control_bridge_20261027_v1_freeze_v1
```

This command only freezes files; main separately authorizes and executes native
physics using its reviewed executor. No source edits after bridge seal; a changed
compiler policy or interface requires a new basename/protocol, not silent edits.
