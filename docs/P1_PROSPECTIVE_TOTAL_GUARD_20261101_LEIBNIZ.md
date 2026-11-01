# Prospective P1 full-vector guard: integration contract

This is an exclusive, opt-in building block for a **new** host. It is not
connected to the sealed structured/CEM hosts and is not authorization for a
confirmation run. The previous `closed_unverified` receipts are unchanged.

The pair must precommit one complete equal cap vector and bind root, task,
goal, checkpoint, action normalizer, hardware, controller, method, source
hashes, and distinct run IDs. `validate_pair` checks common identity and
equal caps, but a digest does not establish when the contract was committed.
The guard must be created before model/goal/source initialization. Every
attributable shared initialization is to be **actually executed and charged
in full independently for each arm**; do not invent a no-op charge or
retroactively assign an unknown cached operation. If re-execution is not
possible, the comparison is not a standalone-full-cost comparison.

`Guard.dispatch(phase, kind, units, callback)` writes and fsyncs the
reservation before the callback. Nested operations have distinct axes, failed
attempts retain reservations, exceeded caps are refused before dispatch, and
any error fail-stops further guarded work. Cleanup needed for safety must
still happen outside the failed guard and be separately recorded; failure is
not a complete run. A pinned journal can be checked by `verify_journal`.

Axes include load calls; actual source/RGB encoder calls and samples; action
encoding; prediction calls and samples; reverse calls and VJP rows; Jacobian,
SVD, solve, candidate score, replay, feedback, physical dispatch, substep,
render and reset. Callers must choose conservative units from actual runtime
shapes/known operation counts. Model encoding and projection are one logical
source-encode call, and prediction and output projection are one prediction;
they are not added as independent calls. Likewise feedback/physical dispatch
may enclose other charged calls: axes describe overlapping work and are never
summed into a scalar.

For a new host, construct the guard before guarded model load, then build
`LeWMFacade` with the real frozen CUDA model. It installs pre-forward
sentinels on encoder, projector, action_encoder, predictor and pred_proj and
offers shape-derived encoding and prediction counters and CUDA-only scalar
gradient routing. The caller must explicitly route setup, score/selection,
structural operations, feedback, every physics dispatch/substep, render,
reset, and failure through the same guard. Do not perform expensive work in
an uncharged callback or count only a placeholder marker.

## Remaining blockers before a confirmation experiment

- A sentinelled `Module.__call__` detects direct module calls that bypass the
  guard, but Python `.forward()`, functional CUDA calls, `autograd.grad`,
  external libraries, and physics calls can bypass module hooks. A callback
  can also hide work. Registered hook consistency is not complete coverage.
- Existing structured and CEM planners require their exact legacy leaf/driver
  types, while the physical host owns a separate meter and loads the model
  before constructing it. Wiring this facade into both real controllers and
  physical backend requires new exclusive implementations or reviewed
  adapters, independent call-graph/physical-boundary coverage checks, and
  tests demonstrating that bypass paths fail. This file does not claim that
  integration has occurred.
- Cap values, bounded wall time, output/storage, CUDA memory/energy, kernel
  work and cost of source verification need an explicit preregistered policy
  and actual enforcement/measurement. Calls, samples, VJP rows, SVD, hardware
  time and memory are different units. Equal vectors are not proof of equal
  FLOPs, latency, energy, or actual complete compute.
- The existing cp10 target was prepared offline. Its model encoding and
  acquisition costs must be charged by the new host or explicitly designated
  external/unknown before any full-cost claim. A cached target with unknown
  upstream cost cannot be made known by hashing it.
- No historical journal is reclassified. `coverage_verified`,
  `full_cost_certified`, and `fairness_proven` remain false. Source hash checks,
  hook events and a hash chain do not prove that all calls were intercepted.

Tests use fake modules only, not CPU model inference:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_prospective_total_guard_20261101_leibniz.py
```
