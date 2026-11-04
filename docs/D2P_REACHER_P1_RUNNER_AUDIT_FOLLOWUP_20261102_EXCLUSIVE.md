# Reacher P1 runner audit follow-up

Scope: source inspection and CPU-only tests, not a physical or neural run.
The P1 runner and P1 preflight changed since the prior saved static audit:
their hashes no longer match that report. The old report is preserved and is
not evidence that the current bytes were sealed before any execution.

Corrections and limits:

1. `run_reacher_d2p_p0_cuda.sh` calls the **P0 Jacobian preflight**, which
   performs executed warmup but does not dispatch candidate plans. The
   separate `run_reacher_d2p_p0_goal_progress.py` does execute candidate
   actions physically. These two P0 programs must not be conflated.
2. P0's `_instrumented_planner` wraps `model.predict` with
   `PredictInstrumentation`. The P1 runner does not call that wrapper or
   instantiate `PredictInstrumentation`/`PredictionBudget`; P1 planner
   receipts include calculated rollout/sample counters, not a complete
   runtime ledger. No claim is made that historical P1 runtime predict calls
   were directly counted by the P0 wrapper.
3. `p0._execute`, reused by P1 suffix and CEM execution, calls `env.step`
   for each requested action but breaks on termination or truncation. It
   returns distances and identity, without a step count or termination flag.
   The P1 gate checking `len(action_witness_raw)` checks the requested witness
   length; it does not independently establish that every step executed.
   A short CPU-only fake-environment test demonstrates this precise behavior.
   This is not evidence that any particular historical run terminated early.
4. P1's first-block `advance_executed_context` has a different policy:
   `_execute_raw_steps` raises on termination or truncation rather than
   returning a shortened success. It shifts observed RGB/action history
   before recompilation; the physical suffix still follows point 3.
5. The source audit does not independently authenticate a specified P0
   evidence file, prove runtime input provenance, or certify the declared gate.
   A preflight/physical run still requires the original strict gate and
   separately verified evidence. No stage admission or equal-budget claim
   follows from these tests.

Run from `le-wm-main` without touching GPU:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_reacher_p1_physical_step_termination_20261102_exclusive.py \
  tests/test_audit_reacher_p1_runner_contract_20261102_exclusive.py \
  tests/test_d2p_model_call_instrumentation.py tests/test_reacher_d2p_context.py
```

The termination test intentionally tests existing behavior, not a fix. A safe
future implementation needs step attempts/returns, termination, and failure
to be recorded durably for every physical branch, and must preserve the
existing gate rather than silently accepting shortened execution.
