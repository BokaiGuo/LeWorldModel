# Direct-H2 zero-logit Jacobian finite differences: MAIN-only CUDA diagnostic

Only NEW source/tests/prepare. Do not edit/re-run sealed direct experiments.
Prepare and tests use saved arrays; ONLY MAIN may call explicit `run` after review.
No neural CPU fallback, new image encoding, AD/Jacobian/VJP, physics or training.
No new Q, optimization, scientific pass threshold or P1 admission.

## Actual inputs, derivative provenance and indexing

Use the TWO new actual direct-calibration tasks, NOT the prior four roots:
`results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01/task_0.json`
and `task_1.json`. External direct report SHA:
`fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4`;
original prepare SHA:
`24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6`.

Roots are respectively
`02a139165f6e8465ec882926c2ceb13a33ca5547f736ed1e3817242902050d06`
and `05844ca2c3ab8ba09846f9084edd71c8365ad74f7ab5961adaa5b0fded6289a7`.
History is source-declared causal padding, NOT observed transitions; diagnostic
results inherit that limitation and cannot upgrade the task's scientific status.

GN `direct_runs[0][0]` is the iteration0 incumbent; its derivative is stored in
`direct_runs[0][1].derivative`, before the first proposal. Prepare independently
requires ZERO20logits and ZERO10x2raw commands, full192x20 finite J, nominal
terminal and task_id identity. The misleading legacy field
`raw_terminal_jacobian` actually contains LOGIT-J. Cross-bind it to the original
`direct_journal.jsonl`: first192 ordered `jacobian_row.raw_control_derivative`,
two `jacobian_forward_packet` contexts, first `jacobian_forward_complete`,
`joint_logit_jacobian` and `derivative_complete`. Independently reconstruct
`Jlogit=Jraw*diag(1-tanh(logits)^2)` ONCE. At ZERO this chain is identity, so
this diagnostic cannot detect every nonzero-point chain bug; report that limit.

No saved-audit/selfchecker green conclusion is read. Original task context and
independent goal are cross-bound to the source native-image journal. Task/model/
normalizer identities, original data hashes, original derivative implementation
closure, new implementation/test/doc hashes and original frozen weights/config
are verified. The full192J is inherited, not newly computed or free pipeline work.
Original direct/image resource totals remain separate inherited fields.

## Fixed design, all32evaluations paid

Time-major20logit coordinates `(packet,primitive_step,xy)` flatten from2x10.
Four directions fixed before results:

1. packet0 first x-coordinate basis e0;
2. packet1 first x-coordinate basis e10;
3. `(e0+e10)/sqrt(2)` equal unit-weight packet direction;
4. NumPy PCG64 seed20261027 standard_normal20, normalized in float64.

Directions are stored once rounded to float32; analytic J@v uses those EXACT
stored coordinates (their norm is recorded, not assumed exactly1after rounding).
Both requested epsilons `[1e-3,1e-2]`, both signs `[-1,+1]`, every root/direction
included without outcome selection. Multiply stored float32 direction by
float32epsilon/sign; NumPy float32 tanh produces actual10x2raw commands. Record
logits, commands and actual float32epsilon. Denominator is2*actual_float32epsilon,
not an unreported alternate coordinate scale.

Each perturbation calls the ORIGINAL source-bound `NativeCuda.rollout` in
`whole_replay` mode, full two packets, from the same saved root context. No
segmented boundary reconstruction, prefix cache, alternate normalization, Q
projection or image intervention. Full packet1 dynamics retains packet0 in the
native rollout. CUDA float32/no_grad/MATH SDPA and TF32disabled match source
direct forwards. Every actual model.predict must enter predictor/action_encoder
exactly once. Reuse the forward implementation, NOT its AD/selfcheck results.

Budget:2roots x4directions x2epsilon x2sign=32wholeH2evaluations;
x2packets=64predict AND64actionencoder,128moduleentries. No zero forward or
extra independent replay: nominal terminal and J are inherited.0encoder,
0existing projector,0AD/J/VJP,0physics,0training. Duplicate inputs still pay.

## Numerical outputs, no passing threshold

`FD=(terminal_plus-terminal_minus)/(2*float32epsilon)` compared against
`savedJ@v` in FULL192space. Save both vectors, L2/Linf/RMSE absolute error,
relativeL2 using `norm(savedJ@v)`, symmetric relative error, cosine, and midpoint
drift from saved nominal. Zero denominators are JSON null, never a fitted floor.
For all8root/direction pairs, also save epsilon stability L2 between the two FD
vectors and its ratio to savedJv norm.16epsilon comparisons; no selecting the
better epsilon or manufacturing a scientific pass/fail threshold.

Large errors could indicate derivative/chain/input/layout/numerical issues, but
also finite-epsilon nonlinearity/float32 cancellation. Four directions are NOT
a full Jacobian proof. Zero-point tanh chain cannot isolate chain errors that
only occur away from zero. A successful run means complete saved measurements,
NOT accurate derivatives, theoretical validity, physical benefit or P1 pass.

## Durable budget and failure handling

Exclusive prepare/run directories. Reserve200MiB plus full4MiB before creating
output; entire output cap4MiB, with128KiB reserved for terminal receipt. All JSON
and journal writes flush/fsync. Predispatch exact packet/module order and64per
resource hard caps; no refund/retry. Module attempts and synchronized returns
are independently durable, so encoder return before predictor failure is not
silently charged as incomplete. Partial native packet contexts and completed
evaluation files survive later scoring/cleanup failure. Raw commands/logits are
durable in evaluation_attempt before native dispatch.

Receipt binds prepareSHA, journalSHA, analysisSHA and all module entries; journal
binds individual evaluation-file hashes. Verify original/new sources and input
bytes after execution. Model beforehash must match original direct saved tensor
hash; afterhash independently rechecked. Unloaded model has model_unchanged=null,
not true. Preserve first execution error ahead of cleanup/report errors; if disk
failure prevents report persistence, print the preserved errors and fail. Hard
kill/disk corruption cannot be promised durable by software. No automatic retry.

## Commands and handoff

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_direct_fd_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_direct_fd_cuda_20261027_v1 prepare \
  --output results/p1_direct_fd_cuda_20261027_v1_prepared01
.venv/bin/python -B -m experiments.spectral_reachability.p1_direct_fd_cuda_20261027_v1 verify \
  --prepared results/p1_direct_fd_cuda_20261027_v1_prepared01 --prepare-sha PREPARE_SHA
# MAIN ONLY, after source/test/prepare review:
.venv/bin/python -B -m experiments.spectral_reachability.p1_direct_fd_cuda_20261027_v1 run \
  --prepared results/p1_direct_fd_cuda_20261027_v1_prepared01 --prepare-sha PREPARE_SHA \
  --output results/p1_direct_fd_cuda_20261027_v1_cuda_main01
```

Mock tests exercise fixed coordinates/tanh, full192error metrics, epsilon
stability, wrong chain counterexample, normalized content/layout refusal,
predispatch caps and partial completion, sync failure, output floor/cap, actual
raw-derivative provenance and saved-only prepare reconstruction. They do not
claim real CUDA FD agreement. This implementation thread runs no actual neural
or physical evaluation. Main owns the new actual CUDA result and interpretation.

Implementation validation2026-10-27:31new array/saved/mock tests passed;
jointly with original direct-screen and saved-direct-audit regressions72passed.
These establish input/implementation contracts, NOT actual FD/J agreement.
No actual CUDA run or new scientific evidence was produced by this thread.
