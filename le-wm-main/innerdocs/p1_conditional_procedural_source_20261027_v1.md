# Bounded procedural source v1 — calibration-only handoff

This is a source initialization protocol, NOT task admission or scientific evidence.
No real reset/render/physics/neural computation was executed by this implementation thread.
Fake-environment tests establish bookkeeping/contracts only; real acceptance remains unmeasured.

## Frozen calibration2 design

Use `innerdocs/p1_conditional_procedural_source_20261027_v1_calibration2_config.json`.
User-specified parent seeds 202761010/202761011; reset spans [202781010,202781014)
and [202781014,202781018); P=2, A=2. All eight attempts execute regardless of
early acceptance; first legal nonexcluded source per slot is retained. No hidden
old factory or rejection while-loop. Recipe/domain are source-design constraints,
not learned-model plateau/attainability thresholds or outcome-tuned admission.
Do not adapt this frozen distribution after observing generated outcomes.

Budget: 8 random resets + 8 explicit placement resets; 8 explicit renders;
zero public env.step/warmup. Real PushT reset invokes two internal physics steps
and one internal goal render: 32 implied physics steps and 24 total render calls
if all operations complete. Actual internal steps are not instrumented; failures
retain attempted/completed counts, not a false exact actual count.
Artifact cap 8MiB; preflight reserves 200MiB plus cap, writes enforce 200MiB floor.

## Main-only commands

Run from `/home/gbk/code/LeWorldModel/le-wm-main`. Preparation is metadata-only.
Choose NEW request/output names; existing files/directories are refused.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_conditional_procedural_source_20261027_v1 --prepare-calibration innerdocs/p1_conditional_procedural_source_20261027_v1_calibration2_config.json --output results/p1_conditional_procedural_source_20261027_v1_calibration2_request.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_conditional_procedural_source_20261027_v1 --run-main results/p1_conditional_procedural_source_20261027_v1_calibration2_request.json --authorize-reset-render --output results/p1_conditional_procedural_source_20261027_v1_calibration2
```

The main-run command ACTUALLY executes simulator resets/rendering. This thread
does not run it. Main owns the environment lifetime. The CLI imports environment
packages only after authorization, plan, known exclusion, disk and closure gates.

`run_main_only` journals operation starts/completions, each attempt, proposal and
slot using hash-chain JSONL and fsync. It retains first error plus secondary errors;
no hidden retry. CLI close failure uses a separate new cleanup sidecar.
Raw source NPZ is no-pickle and contains state/goal provenance, goal RGB,
three copies of ONE observed RGB and raw zero-action padding. This is causal
bootstrap padding, NOT three observed frames or a normalized H2 certificate.

## Remaining gates

Global freshness remains UNKNOWN; calibration exploration accepts scoped unknown
history but never known seed/span/root reuse. Confirmation remains failclosed.
Source hashes bind explicit Python files, not a complete native engine closure.
No normalizer provenance, model encoding, same-Q/full-H2 direct continuous budget
trajectory, legal two-stage attainable witness, or threshold authority is supplied.
Main may independently calibrate thresholds, then freeze separate confirmation;
must not retune against existing physical outcomes. `to_admission_scan` needs main's
actual encoded task IDs; calibration sources cannot become confirmation sources.
Even successful model screening is not real mechanism/efficacy admission.

An earlier 8-slot configuration/request is an unexecuted metadata-only draft. Its
old resolver closure no longer matches and MUST NOT be run. Calibration2 is the
current requested design; its disjoint seeds are not derived from outcomes.
