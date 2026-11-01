# Fixed calibration2 reset/goal diagnostic

This is a MAIN-owned simulator diagnostic, not a scientific physical comparison,
not a proved explanation of P1 failure, and never P1 admission. Agent execution
is limited to pure array/interface fake tests and read-only material validation.

The input is the exact existing calibration2 request and sourcepack, each pinned
by an externally supplied file SHA. The original frozen source closure, full
hash-chain journal, all 24 operation receipts, eight attempt census, both selected
NPZ hashes, root identities and request digests are checked before env creation.
Exclusions are NOT rescanned after generation. No upstream file is copied/edited.

All eight original slot/proposal/attempt combinations run in original order.
For each: original random reset seed, original sample_state using its requested
goal and native polygons, original placement seed/options, one explicit render.
Saved post-reset state is ONLY a comparison reference, NEVER reset input.
Budget is exactly 16 resets, 8 explicit renders, 32 _set_state calls and 16
internal goal renders on success; each setter implies one native physics step
under the sealed original env source. Hooks count setter calls, not engine
internals. Zero public steps, no warmup, no model/encoder/CUDA/neural work.

Instance hooks record requested state7, before/after state7, block linear/angular
velocity, raw and wrapped-angle differences. Internal render hooks record actual
state/hidden velocities and goal request at goal image capture, before/after
render state and exact uint8 RGB byte SHA. Full RGB arrays are not duplicated:
the original selected RGB arrays remain in sealed NPZ; exact array equality is
tested during run. All historical reset summaries and explicit RGB SHAs are
compared for all eight attempts; selected state/goal/RGB/root are additionally
checked against both saved packets. A mismatch does not select/retry/abort a root:
all eight remain the denominator. Runtime exceptions abort without replacement,
retaining pending attempts/arrays in memory report and immediate fsynced journal.
First execution error survives cleanup/source-hash errors. Output report write
failure is returned on stdout with failed status; an unwritable filesystem cannot
be promised to persist an extra final report. Existing journal remains intact.

Artifacts are JSON and hash-chain JSONL only, normally below 1MiB, hard cap 8MiB;
prepare/run require 200MiB free plus full 8MiB reserve; writes preserve 200MiB.
No upstream report/assets copy and no deletion. Exclusive output names only.

## Main commands

From `/home/gbk/code/LeWorldModel/le-wm-main`, obtain file SHAs independently with
`sha256sum` (request FILE hash is not its internal plan_sha256). Prepare is purely
read-only on inputs and creates only the new receipt:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_source_reset_goal_audit_20261027_v1 --request results/p1_conditional_calibration2_main_20261027_v1_request.json --report results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json --request-sha256 REQUEST_FILE_SHA --report-sha256 SOURCEPACK_FILE_SHA --prepare results/p1_source_reset_goal_audit_20261027_v1_prepare01.json
```

Only MAIN after review executes the following, which DOES run native reset physics:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_source_reset_goal_audit_20261027_v1 --run-main results/p1_source_reset_goal_audit_20261027_v1_prepare01.json --prepare-sha256 PREPARE_FILE_SHA --authorize-reset-render --output results/p1_source_reset_goal_audit_20261027_v1_main01
```

Tests: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_source_reset_goal_audit_20261027_v1.py`.
Interpret raw versus angle-wrapped differences separately; equivalent angle
representations are not geometric drift. Nonzero goal hidden velocity or a
snapshot/request discrepancy is diagnostic evidence, not by itself a measured
cause of model failure. Unobserved padding transitions remain unobserved.
