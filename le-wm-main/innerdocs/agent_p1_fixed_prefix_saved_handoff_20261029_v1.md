# Fixed-prefix saved action handoff

This is a saved-only model action selection and validation seam, not a physical
runner or authorization. It rebuilds the frozen host's complete strict consume
and requires exact byte equality with the actual persisted receipt; the host
strict in turn rebuilds its full parent authority. It binds the actual prepare,
report, receipt, artifacts, inherited source/model closure and this three-piece
source. No physical outcome is an input to the selector.
The inherited host full-strict verification may access older physical source
artifacts for provenance; it is separate from `select()` and its outputs are
not candidate-selection inputs. If a policy forbids even reading those sealed
artifacts during authority verification, `prepare` is not authorized until a
separately approved authority design exists; do not silently skip full strict.

All 64 ordered groups and 576 candidates remain accounted for. Within each
group it independently recomputes the frozen conjunction, exact raw chart,
task-Q and `(terminal_Q, iteration)` tie order; it returns exactly the already
selected model witness or an explicit null, never a fallback. It retains task
and source identity for later separate physical guard. All 64 nulls are valid
and result in zero model-witness plans, not a replacement plan. `physical_release`,
`P1_admission` and freshness remain false. The same physical action does not
prove the same full hidden state. New prefix cp5 must be independently guarded;
this seam does not execute or certify that guard.

`select()` is a pure, testable join and assumes its caller has already done
full strict: only `build()`/`prepare` establish this authority. Tests use RAM
synthetic values, including a synthetic model witness, and cannot certify an
actual run. `verify` reconstructs the complete host consume again; it can be
CPU intensive. No worker production full strict, CUDA or physics was run.

From `/home/gbk/code/LeWorldModel/le-wm-main`, MAIN can run:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_saved_handoff_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_saved_handoff_20261029_v1 prepare --host-output /ABS/HOST_OUTPUT --host-report-sha REPORT_SHA --host-prepared /ABS/HOST_PREPARE --host-prepare-sha PREPARE_SHA --host-receipt /ABS/HOST_RECEIPT --host-receipt-sha RECEIPT_SHA --output /ABS/results/agent_p1_fixed_prefix_saved_handoff_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_saved_handoff_20261029_v1 verify --prepared /ABS/results/agent_p1_fixed_prefix_saved_handoff_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_HANDOFF_SHA
```

Output is exclusive, fsynced, bounded, and cannot be overwritten. Complete
physical provenance, recipe/guard, resource budget and physical execution
require a separately reviewed driver; no claim of P1 or fresh roots follows.
