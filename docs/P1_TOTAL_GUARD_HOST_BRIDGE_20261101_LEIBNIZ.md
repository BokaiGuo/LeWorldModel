# P1 total guard host bridge: partial, fail-closed

This exclusive adapter supplies an opt-in process-local wrapper of the actual
legacy `codex_four_arm_durable_20261030.Driver.dispatch` used by both node
planners; it maps homogeneous reservations into the prospective whole-run
guard before the original callback. It also provides guarded model-load,
whole visual-handler, and callback-style physical-meter adapters, plus
structured/CEM draft pair validation. Unsupported physical resources or
mixed reservations fail before dispatch. These functions are not yet wired
into an executable structured/CEM host; no new physical or CUDA run occurred.

The native visual `Sink` calls `meter.begin/end` directly, bypassing a wrapper
around `meter.call`; the guard has no axes for constructor, close, goal bind,
goal setter or state setter. Existing cp10 goal generation/encoding is
offline, and full autograd/functional/physics hook coverage has not been
independently demonstrated. A common full-run vector and wall/memory policy
also remain to be precommitted. `require_host_ready` refuses execution while
these gaps exist. Tests do not establish completeness or fairness.

From `le-wm-main`:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_total_guard_host_bridge_20261101_leibniz.py tests/test_p1_prospective_total_guard_20261101_leibniz.py
```

Six bridge tests and ten guard tests pass. The node tests use the real legacy
durable driver and fake callback, not CPU inference. Original hosts and
historic `closed_unverified` results are untouched.
