# P1 physical callback guard: development contract

This exclusive adapter intercepts an existing `moving.Meter` instance's actual
`call`/`begin`/`end` methods for a scoped physical phase. It executes the
original `Meter.call`, including nested calls from the real legacy
`metered_class`; it does not modify legacy code, its contract, or the
readiness gate. The host must create the meter and enter the scope explicitly:
neither structured nor CEM full-host runner is yet wired to create this scope.

An independently hash-bound, precommitted vector has one attempted-call unit
for each of `constructor`, `reset`, `set_state`, `goal_setter`, `goal_bind`,
`goal_render`, `explicit_render`, `public_step`, `physics_step`, and `close`.
The vector is an additional guard, not a replacement for the total contract.
The existing total axes are charged only for `reset` → `reset_calls`, both
render kinds → `render_calls`, `public_step` → `physical_dispatch_calls`, and
`physics_step` → `physics_steps`. The other resources have no existing total
axis and remain visible only in this independent vector. A public step and
its nested physics steps are distinct dimensions, not interchangeable units.

Each physical reservation is durably recorded before the old meter or actual
callback. A failed callback stays charged. If the total guard or legacy meter
blocks before the physical operation, this reservation is still an attempted
dispatch, **not** proof that physical work occurred; old-meter events and the
total journal provide additional stage information. No internal work or time
within an attempted callback is measured. Hash-chain verification checks the
independent vector and nested operation ordering.

After a failed operation the only permitted callback is an independently
budgeted `close`; this is a cleanup exception, not a successful continuation.
The old meter has no `close` cap: a close counter and its precommitted
independent cap are added to that meter instance only for the scope and
removed afterward. Failed legacy meter calls can leave pending records.
Cleanup itself can fail or be blocked by its cap and must be reported; this
adapter cannot guarantee environment cleanup.

Direct begin/end through the patched instance is rejected. On a clean exit,
meter attempt counts are reconciled with the independent vector, catching
simple pre-captured `begin` calls retrospectively; this is not prevention
before the bypass. Failed scopes do not perform that equality check because a
reservation can precede a blocked old-meter attempt. Captured methods, direct
native environment/space calls, counter tampering, other threads and subprocesses,
and work executed before the scope remain potential bypasses. In particular,
legacy `metered_class.__init__` calls `self.close()` directly on constructor
failure; it is not a `meter.call('close', ...)` and is **not** accounted by
this adapter. Its scope patches begin/end exclusively, so composition with
the visual Sink adapter has not been established. The adapter's lock covers
only this adapter's participants.

Tests exercise the actual legacy Meter and metered-class callbacks around a
fake environment; no GPU or real physics is run. Each resource's callback
failure, caps, nested callbacks, direct and captured-begin bypasses, cleanup,
and unchanged full-host readiness are tested. Until full-host integration,
visual coexistence, constructor-failure cleanup, the other resource axes and
bypass coverage are resolved, preserve `closed_unverified` and fail-closed.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p1_physical_callback_guard_20261101_leibniz.py \
  tests/test_p1_visual_live_host_wiring_20261101_leibniz.py \
  tests/test_p1_total_guard_host_bridge_20261101_leibniz.py
```
