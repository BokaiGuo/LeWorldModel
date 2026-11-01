# P1 physical/visual guard composition: development only

Both existing adapters temporarily intercept the same meter's `begin/end`.
Directly nesting the visual adapter in the physical adapter is incompatible:
the outer physical adapter rejects the Sink's encoder/projector operations.
The exclusive combined adapter retains the physical callback and constructor
close scope, and temporarily gives `begin/end` to the visual adapter only while
the visual boundary executes. It uses the same total guard, meter and store;
it refuses another meter/store, nested physical→visual dispatch, and physical
meter calls attempted during the visual boundary.

For structured, a combined moving proxy is captured by the real cp10
`configure`/`old_episode` path. For CEM, a private copied runner with the same
proxy is passed to the real `execute_one`. Tests reach the actual host
functions with fake environment/controller and fake visual-model hooks, and
stop at the visual boundary. Separately, the CPU integration smoke combines
real source-hash-checked installed PushT, the real legacy Meter and real Sink:
one reset, render and zero-action step; the Sink's encoder/projector hooks
are fakes, not a neural forward pass.

This is not a full host experiment or an end-to-end certification. The smoke
uses temporary pytest journals rather than a sealed experimental report; it
does not exercise the real CUDA visual Handler or planner. The installed PushT
file is hash-checked in the test but is not included in that test's total
guard source contract. Real CUDA plus live RGB/physical composition, complete
source/dependency binding and the common full-resource contract remain open.
The existing public full-host readiness function remains unconditionally
fail-closed; legacy host entry points remain outside this development adapter.

The method-swap and structured configuration patches are process-local, not
security isolation. Locks cover only cooperating adapters; captured methods,
direct native/functional calls, unrelated threads and subprocesses can bypass
them. Physical constructor failure cleanup and zero-close budget limitations
from the constructor adapter remain. Do not infer fairness, full cost coverage,
or P1 admission from these tests.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy timeout 60s \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p1_combined_guard_wiring_20261101_leibniz.py \
  tests/test_p1_visual_live_host_wiring_20261101_leibniz.py \
  tests/test_p1_constructor_close_guard_20261101_leibniz.py \
  tests/test_p1_physical_callback_guard_20261101_leibniz.py \
  tests/test_p1_visual_sink_guard_20261101_leibniz.py
```
