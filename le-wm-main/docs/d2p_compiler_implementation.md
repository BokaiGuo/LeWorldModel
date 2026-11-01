# D2P reusable implementation — 2026-11-02

This is an implementation checkpoint, not complete research closure.
Existing sealed experiments and stage admission rules are unchanged.

## Implemented path

`d2p_local_compiler.LocalCompiler` provides task-weighted finite-horizon
geometry, rank-truncated unreachable residuals, action-scaled damped progress
programs, signed spectral prefixes, successor relinearization, and original
`stage + beta * local_ridge_reach` selection. Every candidate retains its
complete action sequence and full latent/action successor context. Full and
split rollout contexts must agree exactly.

Its line search always includes zero. Local ridge values remain unconstrained
approximations; they are not physical reachability certificates. Compiler
operation caps include failed attempted backend calls and are not neural
sample or wall-clock budgets.

`d2p_pusht_cuda_backend.PushTCudaBackend` connects this compiler to the existing
frozen PushT model: three latent boundaries, two intervening normalized action
packets, five raw actions per packet. CPU neural execution is rejected.
`observe` encodes actual RGB and appends actual executed actions without a
model prediction. Public normalization constants remain a declared convention,
not newly certified checkpoint training statistics.

`d2p_feedback_controller.run_compiled_feedback` retains the complete selected
plan, dispatches its first packet, then replans from `observe` output. Prediction
records never replace real feedback. Native partial stops and failures preserve
reported action counts; unknown execution counts are explicitly marked.
Callbacks remain a trust boundary: field filtering alone cannot authenticate
RGB, exclude hidden simulator access, or certify compute usage.

Example integration:

```python
result = run_compiled_feedback(
    initial_context,
    compiler_factory=lambda context: compiler,
    execute=authenticated_simulator_packet,
    observe=backend.observe,
    packet_size=5,
    max_replans=2,
    event_sink=durable_event_writer,
)
```

Reuse one compiler to retain cumulative operation caps. A factory constructing
new compilers must enforce a separate shared total budget. The example requires
caller-provided execution and durable-event adapters; it is not itself a tested
live simulator entry point.

## Verification

CPU tests:

```bash
.venv/bin/pytest -q tests/test_d2p_local_compiler.py \
  tests/test_d2p_feedback_controller.py tests/test_d2p_pusht_cuda_backend.py
```

Host-CUDA engineering smoke:

```bash
.venv/bin/python -m experiments.spectral_reachability.d2p_compiler_cuda_smoke \
  --inputs results/p1_dev303_20261101_codex_blind_inputs.npz \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/NEW_EXCLUSIVE_REPORT.json
```

Completed output: `results/d2p_compiler_cuda_smoke_20261102_01.json`.
Four programs, 23 returned rollouts, five returned Jacobians, exact independent
replay, unchanged model. Known development input only; no physical steps.
The subsequently added RGB `observe` method has not yet received live-CUDA
feedback-loop acceptance. No claim of certified total compute is made.

## Eight-way existing P1 tail execution

The separate existing development planner's four prefixes × two metric variants
were executed from their exact recorded cp5 RGB/action states.
Two pre-execution bugs were corrected in the previously unsealed runner:
metric-pair rows incorrectly treated as unique per-root rows, and an invalid
array-as-boolean prefix check. Prefix, tail and cp5 hashes now bind explicitly.

Output:
`results/p1_dev303_dynamics_cp5_tail_physical_20261102_exclusive_RUN_MAIN01/report.json`

Report SHA256:
`a1212d48d6aac973cdc3f79fc0c7b428aac0aff2365d2e72187e4af2a1c67202`.

All eight simulator branches completed ten tail steps with no native stop or
reported error. Selected terminal native rewards were -303.2803 / -299.1080
(latent / tangent); greedy was -291.5349 / -285.1977. These are development
results, not success, mechanism evidence, independent roots, or equal-total-cost
certification. They are not execution results of the new reusable compiler.

## Still required for the full proposal

- Live simulator acceptance of the reusable compiler + real-RGB observer loop,
  including old-plan-tail rescoring/replacement and authenticated event storage.
- General multi-program search regenerating programs at successors, not only
  the implemented two-program witnesses.
- Dynamics-driven horizon/search scheduling connected to this compiler and a
  shared enforced resource ledger; separate P3 engineering assets are not this
  integration.
- PointMaze/Reacher backend adapters and task/action-mask contracts.
- Unified resumable E1/E2/E3 execution, applicable strong-baseline adapters,
  complete accounting and failure/censoring handling.
- Frozen independent-root confirmation and scientific P1 mechanism/performance
  gates. Tests and the CUDA smoke do not open these gates.
