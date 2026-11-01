# P3 offline quality / resource audit v2

Engineering calibration only. P1 is unpassed; neither P1 nor P3 is admitted.
This analyzer imports only Python's standard library. It reads saved JSON,
journals and image bytes, never imports a model/simulator, runs training,
readouts, predictions, physical rollouts, threshold selection or root selection.

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_quality_cost_audit_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_quality_cost_audit_20261027_v2
```

The three requested reports anchor the audit. It follows their explicit saved
runtime/protocol bindings, verifies upstream source manifests against both live
and archived bytes, verifies selection content hashes, joins by full root SHA,
method and repetition, and rejects duplicate/missing/extra identities. Physical
journals and their RGB hashes are checked; no image decoding is needed.
External checkpoint/config bindings are disclosed but not rehashed or loaded.
Consumed evidence is hashed again before and after writing. Output is exclusive:
`le-wm-main/results/p3_quality_cost_audit_20261027_v2/`. A second launch refuses
overwrite. Analyzer, test and this document are archived there with hashes.

The generated `report.json` preserves every physical execution, guard abort,
failure, pose metric and officialstate metric. `report.md` renders per-execution
resource/quality and matching tables; `manifest.json` binds generated artifacts.
Search maximum depth is distinct from selected plan depth and actual simulator
returns. Partial-token equivalents are descriptive step/5 arithmetic, not proof
of complete token execution. The phase named `evaluation` in the saved receipt
is the selected-path predictor verification, not a new neural evaluation.

Resource matching is explicit by dimension: search horizon, selected horizon,
planning/verification/total predictor samples, saved resource vector, physical
attempts and actual endpoint steps. A matched partial guard endpoint is not
promoted to a completed horizon. Full-path matching requires both completions.
Saved runtime action/input backward calls and VJP counts are reported separately;
the absence of a separate dense Jacobian counter is represented by null, not zero.
No inference about upstream compiler costs is made.

Two executions of a saved root-arm share a single planning receipt. Planning
and verification totals are charged once per root-arm; physical counters cover
both replays. Repeats are dependent reproducibility records, not independent
samples. No repeat confidence intervals are computed. Equal saved counters do
not establish total-FLOPs equality: compiler, model setup, reset/warmup and other
upstream work are outside this ledger. Quality deltas are descriptive only;
no adaptive superiority or P3 admission is claimed.
