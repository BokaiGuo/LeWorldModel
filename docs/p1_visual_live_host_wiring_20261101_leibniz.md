# P1 visual live-host wiring: development contract

The exclusive wiring module is a development integration, not an executable
full-cost experiment. Its public `run_structured` and `run_cem` call the
existing unconditional readiness refusal; neither admits a host run.

The structured private scope temporarily installs a guarded clone of the
original actual RGB episode before calling the real cp10 `configure`. It
checks the common wrapper captured that exact function in `old_episode`, and
restores modified host and RGB attributes on normal and exceptional exit.
Concurrent entry through this scope and unexpected preconfigured episodes are
rejected. This scope mutates shared Python module attributes temporarily: the
lock protects only participants using this module, not unrelated threads or
arbitrary mutation, so it is not a process-wide security boundary.

The CEM private helper rejects a different original moving module, constructs
a local guarded runner, verifies its guard and isolation, and passes that
runner to the actual `execute_one`. The helper itself is for testing only and
does not establish other resource coverage. In particular, calling legacy
entry points directly, or invoking private Python helpers, is not prevented by
this module; only its public host entry points are fail-closed.

The tests invoke the actual structured configuration and episode body and
actual CEM `execute_one` through to their visual boundary. Their environment,
controller, meter and guard boundary are test doubles; the boundary stops
before any physics or neural work. They additionally reject wrong structured
capture, preconfigured/concurrent structured use, unexpected CEM moving
source, and a CEM proxy bypass. No actual full host, live RGB, GPU, or physics
run is claimed. The previous saved-RGB CUDA probe is distinct evidence.

The new module and test are not included in any historical source manifest.
Any future executable host needs a new source-bound preparation and fixed
common budget, guarded model/goal initialization, actual node and physics
dispatch, failure accounting and bypass tests. Preserve the existing readiness
gate and `closed_unverified` until all relevant costs are independently
covered; this work supplies no fairness or P1 admission.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p1_visual_live_host_wiring_20261101_leibniz.py \
  tests/test_p1_visual_guard_host_20261101_leibniz.py \
  tests/test_p1_visual_sink_guard_20261101_leibniz.py
```
