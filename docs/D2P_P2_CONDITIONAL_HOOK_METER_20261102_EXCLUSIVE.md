# Conditional composition hook meter: host integration

`p2_conditional_hook_meter_20261102_exclusive.RoutedModelMeter(ledger)` routes
both conditional compilation and common-objective operations to the same
durable ledger. Its public interface supplies `phase(category)`,
`call(axis, operation)`, `counts` (a detached snapshot of attempted counts),
`returned_counts`, `ledger` (a routed ledger view), and `finish()`. Counts use
full ledger-category keys, not the older arm-only keys. `RoutedMeter` is an
alias. `call` accepts the four model-hook axes and the explicitly permitted
operation axes; the routed ledger view accepts only operation axes.

For a future source-bound host runner, after acquiring the cooperative GPU
lock and authenticating the frozen model, inputs and output directory, the
intended integration inside its guarded native model session is:

```python
meter = routing.RoutedModelMeter(journal)
with objective.native_model_scope(model, meter):
    result = bridge.run_bridge(fixed, transition, meter.ledger, device)
meter_summary = meter.finish()
```

`transition` must be the authenticated frozen model transition on the
authorized device. Pass `meter.ledger`, **not the underlying raw journal**, to
the existing bridge, so compilation operation attempts are also capped.
The existing `native_session` installs an older common-only hook meter:
do not stack it with this scope. A host implementation must establish an
equivalent source-bound, locked model session without those duplicate hooks.
The caller must close the journal and retain its partial data on failure;
success also requires an independently checked ledger and result binding.
This snippet is not a standalone execution command or authorization.

Limits, charged on attempted calls before dispatch:

| Category | Operation | Maximum / required for completion |
|---|---|---:|
| composition_inherited | prefix transition | 16 |
| composition_inherited | suffix transition | 32 |
| composition_inherited | 192-output Jacobian | 16 |
| composition_inherited | bounded ridge solve | 16 |
| composition_inherited | each of action_encoder, predict, predictor, pred_proj | 48 |
| common_composition | transition / autograd gradient | 288 / 128 |
| common_no_composition_sra | transition / autograd gradient | 288 / 128 |
| each common category | each model-hook axis | 288 |

These are caps for the specified 16-candidate schedule, not an equal-total
budget. A native model whose actual hook multiplicity differs will fail
closed at the limit or at `finish()`; do not silently adjust the caps.
The counter does not measure CUDA kernels, hidden backward engine operations,
FLOPs, energy, I/O, model load or prior bank generation. An output-Jacobian
call is counted as one operation, not as independently measured 192 VJPs.

The ledger writes attempt before operation, then return or failure; unknown
routes and over-cap attempts write a rejection and latch failure, and the
meter refuses further dispatch. A caught nested dispatch failure is also
latched and recorded. `finish()` requires the exact successful census and
records incomplete schedules. A phase exception records its exception type.
Failures of ledger I/O itself can prevent durable failure reporting and must
invalidate the run. Only single-owner-thread use is supported; a wrong-thread
call latches and raises without trying to write to the non-thread-safe
journal. The API is not a security boundary against a caller bypassing the
view or invoking unhooked model entrypoints, and does not authenticate the
model, source closure, device or GPU lock.

CPU validation command from `le-wm-main`:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_conditional_hook_meter_20261102_exclusive.py tests/test_p2_same_bank_conditional_suffix_ablation_20261102_exclusive.py tests/test_p2_same_bank_native_objective_20261102_exclusive.py
```

Actual result: 20 passed. The full 16-candidate schedule uses real saved
candidate bytes and a synthetic CPU model; its model entrypoints go through
the actual hook wrapper and the journal is independently chain-checked. Tests
also cover cap rejection, model and phase failures, nested caught failure,
hook restoration, bad routes, objective API, and snapshot isolation. This is
not evidence of a real-model or physical result.

Meter source SHA-256:
`717f5587048a78f44e840903238786244ba7f4fe27d303b7c09d597ee00cf42a`.
Test source SHA-256:
`eb4d33efa7a5c2cd04533972a35860eaa52d039ae8ae4c683df69dc7f3fd91e1`.
These are development snapshot hashes; a future host plan must bind its
own complete explicit source list and preserve known source-drift failures.
