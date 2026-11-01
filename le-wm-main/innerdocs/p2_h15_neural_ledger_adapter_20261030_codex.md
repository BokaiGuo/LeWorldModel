# P2 H15 prospective neural ledger adapter

## Scope

This is a new opt-in facade over `p2_durable_hook_driver_20261030_codex.Driver`,
shared with `p2_h15_cpu_metered_backend_20261030_codex.Backend`. It provides
the `scope`, `events`, `call`, `caps`, `attempted`, `completed` and `receipt`
attributes used by the existing `NativeSignal` / `NativeTree` call sites. It
does not instantiate, launch or authorize those neural backends.

`create` validates the complete native task, canonical contract digest and
the task's root, task, model and action-domain bindings before creating an
exclusive driver directory. Its hardware and meter identifiers are only
declared contract fields: this adapter does not attest actual hardware,
same-consumed model bytes, runtime device, or precommitment time. Every
neural hook scope binds the native-signal, native-tree, CPU-backend, adapter
and task hashes; direct source drift or task mutation stops subsequent calls.
The facade returns event/counter copies; the `events` copy required by the
native ABI may add substantial overhead for large journals and has not been
profiled on the host.

The same native ledger records `predict` API/sample/context dimensions,
action encoder, predictor and projection shapes, Jacobian requests, VJP
directions, grad calls, raw-backward callbacks and other explicitly hooked
operations, using the existing metadata validation and nonrefundable cap
logic. The P2 contract directly projects predict samples and VJP directions;
the CPU backend projects array SVD, solve, and its dedicated selection-metric
cap. Other native caps are inherited from the caller's complete original cap
vector, not inferred from the P2 contract. Hooked events share one ordered
durable journal, including nested CPU calls; overlapping API and module axes
must not be summed as independent work.

## Verification and limitations

The CPU test reissues the actual saved root-0 H15 signal resource-event tree
through the existing `replay_resources` function, then compiles and scores the
real saved arrays with the metered CPU backend on the *same* driver. It checks
6 hooked prediction samples, 192 VJP directions, 2 SVDs, 1 solve and 40 scores,
the recorded winner, durable journal replay and projected caps. The reissued
neural tree is **inert saved metadata**, not model inference or independent
verification of the original model's execution. Separate CPU tests execute
the original `NativeSignal.call` and `observe` methods with NumPy/module stubs,
checking IO indices, scopes, hook cleanup, refusal and failed callback
accounting. These fixtures are not a CUDA backend or model-fidelity test.

Sealed H15/CEM host code requires exact `core.Ledger` type and original fixed
caps; it cannot accept this facade unchanged. A new owned runner and backend
interface are required. Real IO tensor journaling must be integrated with the
durable resource journal and independently checked; the facade does not bind
or validate those IO events. Full model/checkpoint same-consumed-byte binding,
native tensor shape/device checks, hardware and precision, all hidden calls,
inherited work, physical substeps, synchronized whole-run wall/device peak,
and FLOPs remain unverified. This cannot produce a complete P2 resource
receipt, certify a total budget, imply equal computation, or authorize P1/P2.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_h15_neural_ledger_adapter_20261030_codex.py \
  tests/test_p2_h15_cpu_metered_backend_20261030_codex.py \
  tests/test_p2_durable_hook_driver_20261030_codex.py \
  tests/test_p2_hook_contract_bridge_20261030_codex.py \
  tests/test_p2_preregistered_resource_vector_20261030_codex.py \
  tests/test_p2_saved_action_fairness_20261030_codex.py
```
