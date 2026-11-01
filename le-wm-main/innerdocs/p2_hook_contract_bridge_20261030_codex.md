# P2 hook-to-contract bridge: preparation only

## Existing hooks inspected

- `agent_p3_runtime_contract_20261028_v1.Ledger.call` computes resource deltas
  from shape/direction metadata, checks caps before the callback, emits attempt
  before dispatch and return/failure afterwards, and leaves failures charged.
  Its receipt exposes attempts, completions, events and nested inclusive wall.
  `audit_ledger` re-derives deltas and totals; unknown FLOPs stay null.
- H15 `p2_h15_host_lifecycle_20261029_v1` constructs that ledger directly.
  Its `GuardedNative` requires `type(ledger) is core.Ledger` and the original
  `single.caps()`; the CEM host has analogous exact checks. Therefore this
  opt-in bridge is **not** directly injectable into those sealed host runners.
  A separately owned, reviewed caller and backend interface are needed; no
  monkeypatch, sealed change, or silent relaxation is authorized.
- In `p2_h15_sra_host_candidate_20261029_v1`, spectral bank construction,
  selection scoring and outer replay calls occur outside explicit corresponding
  ledger calls. A recorded zero for SVD/solve/replay on that ledger is not proof
  of zero standalone work.

## What the bridge does

`HookBridge` implements the scoped `call`, `scope`, `attempted`, `completed`,
and `receipt` interface by delegating to a new instance of the existing public
ledger. It checks the canonical external contract digest and tightens (never
loosens) the existing caps for five same-unit axes: predict samples, VJP
directions, array SVD calls, solve calls, and replay calls. The underlying
ledger's explicit sink is required, and an attempt-sink failure prevents the
callback. The sink may be an in-memory function: durability is not established
by this bridge. Calls bypassing the ledger are not constrained by it.

The diagnostic independently runs the existing ledger recount and reports
**hook-observed** counts with `coverage=unknown`. It does not generate a
complete P2 contract receipt. Selection evaluations, FLOPs, selected physical
substeps, full standalone planning wall and device peak remain unmapped; even
the five mapped axes lack a coverage proof. Ledger lifetime wall is not whole
standalone wall. No measured zero is inferred for inherited work.

## Saved audit and test boundary

Running the module without arguments reads the pinned H15 parent report SHA and
its two bound child report SHAs, then recounts each embedded native ledger.
Each root has 132 predict samples and 192 VJP directions. The zero reported
hook counts for SVD/solve/replay are *not* zero-work claims. This is not a
complete upstream or physical-journal revalidation.

CPU fixtures test real public `Ledger.call` delegation, all five mapped
pre-dispatch caps, failure/no refund, attempt-sink failure, tampered event
recount, contract changes and old exact-type incompatibility. No fixture
certifies a real runtime. Before a future real runner may claim complete
resources, it needs an independently precommitted contract, a new caller that
actually connects all work sites and inherited charges, durable failure-safe
event journals, separate FLOP/whole-wall/device-peak and physical counters,
and independently checked coverage and consumed-byte bindings. None is
authorized by this adapter; no fairness, P1 or P2 admission follows.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_hook_contract_bridge_20261030_codex.py \
  tests/test_p2_preregistered_resource_vector_20261030_codex.py
```
