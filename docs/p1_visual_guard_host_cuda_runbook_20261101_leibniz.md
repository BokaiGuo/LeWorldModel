# P1 visual guard: corrected host CUDA runbook

This supplements the source-bound
`p1_visual_guard_host_20261101_leibniz.md` without changing it. The previous
assistant message incorrectly used relative paths. The implementation requires
absolute paths for both the prepared input and output directory.

From `/home/gbk/code/LeWorldModel/le-wm-main`, create a **new** preparation
file; choose a name that does not exist. `prepare` prints its SHA-256:

```sh
.venv/bin/python -m experiments.spectral_reachability.p1_visual_guard_host_20261101_leibniz prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_visual_guard_host_20261101_leibniz_PREP_NEW01.json
```

Replace `<PREPARE_SHA256>` with that printed hash and use a fresh output
directory. The cooperative GPU lock is acquired by the probe.

```sh
.venv/bin/python -m experiments.spectral_reachability.p1_visual_guard_host_20261101_leibniz probe \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_visual_guard_host_20261101_leibniz_PREP_NEW01.json \
  --sha256 <PREPARE_SHA256> \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_visual_guard_host_20261101_leibniz_RUN_NEW01 \
  --authorization AUTHORIZE_P1_SAVED_VISUAL_PROBE_CUDA \
  --authorize-host-cuda
```

The existing `PREP_LEIBNIZ01` and `RUN_LEIBNIZ01` are historical artifacts,
not names to reuse. A changed bound source requires a fresh preparation; do not
rewrite an existing preparation or output.

## Scope and remaining host wiring

The actual structured `rgb.episode` looks up `moving.encode_boundary` in its
globals. The common structured host captures `old_episode` before installing
its wrapper, which subsequently calls that captured function. A complete new
host must arrange for the captured function to be the guarded
`structured_episode(guard)` before that capture, with scoped restoration.
Merely constructing the adapter does not install it.

The actual CEM `execute_one` takes `moving` from `runner.rgb.moving` and calls
it from `encode_current`. A complete new host must pass
`cem_runner(guard, runner)` into that execution path and verify that the
proxied runner is the one actually consumed. Merely constructing it does not
run the controller.

The saved-RGB probe calls the real visual Handler and Sink with the guarded
boundary on CUDA; it does **not** execute either host, a live observation,
physics, or planner. Hook and journal tests do not independently prove that
every alternative neural, functional, captured-method, physics, or failure
path is guarded. The existing full-host readiness gate remains unconditionally
closed, including the legacy visual-Sink gap entry; this probe is not a reason
to remove it. Model/goal initialization and provenance, constructor/close,
goal/state setters, upstream goal generation, physical and node dispatch,
candidate selection, autograd and failed attempts, and common full-run
resource/time/memory limits need end-to-end closure and independent bypass
tests before reconsidering the gate. No full cost or fairness certification
follows from this runbook.
