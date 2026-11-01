# Prospective native C1 CUDA compiler v1

New runner only; no physics entry point, training, CPU neural fallback, old-source
edits or P5 admission. Main executes CUDA after review. The bridge/executor are
independently owned; no messaging tool was available, so this is the coordination
contract for main/Ohm. Independent goal spec now supplied by main:
`results/p5_native_goal_scene_20261027_v1_scene_main_v1/goal_spec.json`.
The runner does not manufacture a goal or authorize physical execution.

## Exact bridge contract

Use `p5_prospective_control_bridge_20261027_v1` ready protocol, independently
retaining its **canonical `bridge.digest(protocol)`** (not file SHA). The compiler
source must equal this runner's absolute path and SHA in `runner_sources.compiler`.
Numerical input is ONLY `bridge.compiler_bundle`: native RGB/proprio frames0/5/10,
two history env packets and separately declared goalRGB. Exactly seven data files.
No factual frame15/state/evaluator or fidelity-prediction input. The bridge itself
authenticates factual provenance including historical diagnostics; those numeric
diagnostics are not passed into the CUDA core or used to choose anything.

Preparation rehashes frozen v8 seals, but never invokes v8 archived-data
`prepare/verify_manifest/validate_cuda`, v11 factual loader or dataset row readers.
Only four v8 manifest fields are carried into `v8.load_cuda`: seals,
official_weight_mapping, expected_dynamics_schema, expected_model_receipts.
Model construction/weight loading remains frozen native v8 CUDA-only, offline.
All modules frozen/eval; before/after receipts must equal the bridge identity.
The fresh neural process must have no physics PYTHONPATH or preimported dinov2.

Fixed seed0, TF32 off; **SDPA MATH for encoding, nominal, every JVP, ridge nonlinear
evaluation and selected replay**. One four-image encoder batch; one three-context
proprio call. Constants are created under no_grad, not inference_mode, to support
double-backward JVP. No FD forwards or additional candidates.

Successful attempted and completed counts must BOTH equal:

| Operation | Count |
| --- | ---: |
| official_encoder / RGB frames | 1 / 4 |
| proprio_encoder | 1 |
| predictor | 13 |
| JVP columns | 10 |
| autograd.grad | 20 |

Predictor phases: nominal1 + ten sequential strict JVP primals10 + ridge1 +
selected replay1, including replay when nominal wins. JVP uses existing
`functional.jvp(strict=True, create_graph=False)` double backward, not forward AD.
No CPU neural tests are permitted or included. Counts are runtime gates, not
assumed cost receipts; unsupported derivatives abort rather than fabricate counts.
The v8 4GiB post-load free-scratch policy is retained, not a measured peak bound.

## Coordinate and arithmetic interface for the independent array audit

Source literal mean/std extracted with AST, without dataset/torch import. Native
u is NOT divided by100. The dataset projection is exactly:

```
pixel32 = (100*u64).astype(float32)
u32 = pixel32/float32(100)
z32 = (u32-mean32)/std32
```

Nominal is five exact native `[0,0]` commands, NOT normalized zero. Preserve
history two packets and actual three-context proprio. JVP coordinates are the
ten time-major normalized outgoing entries around `z0=normalize(native_zero)`.
Residual is independent goal minus nominal, D=98304:
`G=J.T@J/D`, `b=J.T@residual/D`, `delta=solve(G+.01*I,b)` on CUDA float64.
No tuning, restarts, physical scoring, finite differences, line search or damping
change. Candidates are nominal then one full ridge step; stable global float64
goal-feature MSE argmin, nominal-first exact tie.

The centered inverse is explicitly frozen as
`ridge_native_u64 = delta64.reshape(5,2)*float64(source_std32)`.
This centers at exact native zero, avoiding the tiny native offset caused by
evaluating rounded `z0*std+mean`. It is not a projection or step scaling. Reapply
the dataset float32 projection to this actual native packet for ridge nonlinear
evaluation and selected replay. Retain both ideal `z0+delta` and effective
normalized action; their finite-precision mismatch is disclosed, not treated as
exact coordinate equivalence. Any nonfinite or |u|>1 ridge candidate fails the
ENTIRE compilation, even if nominal would win; never clip or fall back. This
support is bridge policy, not native environment's advertised Box.

Receipt `action_coordinates` numeric fields:

- `action_mean_float32`, `action_std_float32`: [2], exact values of source float32.
- `nominal_native_u64`, `nominal_normalized_f32`: [5,2].
- `history_native_u64`, `history_normalized_f32`: [2,5,2].
- `ridge_native_u64`, `ridge_normalized_f32`, `ridge_ideal_normalized_f64`: [5,2].
- `ridge_roundtrip_max_abs`: actual-vs-ideal normalized max abs (also top-level).

If the ridge proposal is inadmissible, save its raw array before rejecting;
later numeric fields can be absent. Nonfinite proposals are stored in NPY, never
serialized as invalid JSON. All candidate costs retained in fixed-order
`candidates=[{id, env_packet, independent_goal_mse}, ...]` as each returns.

The **exact eight** bridge `artifact_roles` (absolute paths) are:

| Role | Shape | Dtype |
| --- | --- | --- |
| response_columns | [98304,10] | float32 |
| goal_features | [1,256,384] | float32 |
| nominal_successor | [1,256,384] | float32 |
| ridge_successor | [1,256,384] | float32 |
| selected_replay | [1,256,384] | float32 |
| gram | [10,10] | float64 |
| rhs | [10] | float64 |
| ridge_delta | [10] | float64 |

Each role points into `artifact_bindings={absolute_file_path:file_sha256}`.
`arrays` additionally gives shape/dtype/path/fileSHA. Extra raw outputs are also
bound: encoder dictionary tensors (including all4 patch arrays), proprio output,
all tensor outputs from every predictor tuple (full three-context output, not
only last successor), ten JVP primals and columns, residual, normalized inputs,
native proposal and effective normalized ridge. No object arrays/pickle.
Selected successor must satisfy `torch.equal` to the corresponding search output,
using the exact same normalized packet, context, weights and MATH backend.

## Durability, failures and scientific limits

Every module dispatch has pre-hook attempted and post-hook completed records.
Grad and JVP attempts are recorded before dispatch too. Numeric outputs are
saved immediately on return, before shape/finite gates. Every array and journal
event is fsynced, including output directory; immutable array filenames cannot
be overwritten. Hash-chained `journal.jsonl` preserves ordered intents, counts,
returned raw outputs, candidate costs and selection. Interrupted work stays
unknown; no truncated journal authorizes physics. A hard process kill can leave
no terminal report; main must retain that directory, never reuse it.

Ordinary exceptions produce `compiler_failed` receipt and CLI exit1. Final model,
source/allowed-data, artifact and hook-removal checks are independently protected;
an original error is never replaced by a later source/model cleanup error.
Failure receipts retain actual partial arrays/counts/costs. No selected packet
authority on failed counts, unsupported derivatives, replay mismatch, changed
weights/sources or incomplete work. Returned counts mean Python API completions;
internal work of a failed CUDA call cannot be inferred, so failure keeps
`partial_work_unknown=true`. Journal/storage failure itself may prevent terminal
receipt persistence; preserve the durable prefix and do not retry in that output.

Only a complete receipt gets schema `bridge.TAG+'.compiler'`,
status `selected_packet_replayed`, partial_work_unknown=false and exact bridge
counts. This is merely a compiler interface gate. P5=false, C2=false. Goal-feature
MSE is mixed-scene proxy, not object progress. No new physical action is executed
by this runner. Main must run bridge `freeze` to durably commit all three branch
slots BEFORE constructing ANY physical branch env; executor is separate.

## Main commands (fresh outputs only)

Working directory `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_prospective_c1_cuda_compiler_20261027_v1.py
```

First Ohm/main creates a NEW bridge ready prepare using the actual independent
goal spec above, this absolute compiler source, and the independently reviewed
executor source. The earlier blocked bridge protocol must NOT be patched.
Then substitute the ready protocol and independently retained canonical digest:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p5_prospective_c1_cuda_compiler_20261027_v1 prepare --bridge-protocol /ABS/NEW_BRIDGE_READY/protocol.json --bridge-sha256 CANONICAL_BRIDGE_DIGEST --output results/p5_prospective_c1_cuda_compiler_20261027_v1_ready_main_v1
```

Retain FILE SHA256 of its new manifest; main-only CUDA command:

```sh
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_prospective_c1_cuda_compiler_20261027_v1 run-cuda --manifest results/p5_prospective_c1_cuda_compiler_20261027_v1_ready_main_v1/manifest_frozen.json --manifest-sha256 MANIFEST_FILE_SHA256 --output results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1
```

Main then independently array-audits and runs bridge freeze with this output's
`compiler_receipt.json`. Physics is main-only, after freeze validation.
This agent executes only pure non-neural tests and a blocked preparation against
the existing blocked bridge protocol to seal reviewed files; that manifest does
NOT become ready merely because a goal later appears. A new ready bridge protocol
requires a new compiler preparation, never modifying any frozen source/manifest.
