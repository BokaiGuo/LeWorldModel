# P3 action-Jacobian v3: bounded saved-outcome audit

New audit only; all scheduler/consumer source and actual outputs remain sealed.
No environment, physics, CUDA, model, neural CPU, NumPy, SVD, download or rerun.
Uses standard-library arithmetic and fixed-size Pillow PNG decoding only.

Inputs are the exact actual v3 physical report (SHA256
`d0f29d41ab0af04eac2c50e33d24385278868b087fb1002ab9f89a7583f044e8`),
physical prepare (`5cb7ef6cbebf38859dbf1b087131c3dca8a9b11cedad26cf64978c4416df2a6d`),
and the v3 scheduler prepare/report; no v2 outcomes replace these records.
All source/input/archive bytes and per-arm files are rechecked before/after.
External weight/config bindings are retained, not rehashed or loaded by this audit.

## Independent checks

Full SHA root+method joins (never array order), exactly four calibration roots,
eight distinct arms and one repetition. Same commands do not merge executions.
Each selected path is independently reconstructed from three integer indices in
the original full16bank. Original task goal/Q, root context, model identity,
147 predict samples (144 planning+3 verification), action encoder and token counts
are checked. Planning per-arm JSON/events are bound; saved selected model Q-cost
is recomputed with bounded 192x192 arithmetic. No Jacobian/SVD is recomputed.

Each fixed 224x224 RGB PNG is hash-bound and its exact RoyalBlue pixels counted.
Journal order is checked as initial -> attempt -> returned -> observation, with
a fresh visible, nonterminated RGB required before every dispatch. Raw command
prefix, all attempts/returns, image/state snapshots and original guard receipt
must agree. Root history/goal RGB hashes match source references and paired reset
contexts must agree. Freshness/full-H3 classification is recomputed, not trusted.
Pose and installed-official-formula labels are independently reconstructed from
saved state+goal; the official metric deliberately retains its velocity/agent
coordinates and its original angle formula. The object-pose diagnostic is distinct.

All partials, guard failures, trace/image references and original encoding receipts
remain in output. Encoder vectors are checked for shape/finiteness and their Q-cost
is computed arithmetically; this does not independently reproduce neural outputs.
Actual encode calls/samples/failed calls are reconciled with completed/attempted
receipts. Source seals attest unchanged bytes, not execution authenticity.

## Comparison and denominators

Only two complete, same-root/same-context 15-step H3 branches form an eligible
pair. Same-step fresh partials remain guard-partial; all other pairings unmatched.
No partial comparative delta enters the new audit, even when endpoint values tie.
Its per-branch endpoint labels remain explicitly descriptive. No tuned tolerance,
selection by opened labels, replacement root, deduplication or missing-arm dropping.

Metrics are reported separately: pose cost, official position/state distance,
object position/angle, official/object-only success, actual terminal encoding Q-cost
and selected model Q-cost. Every delta is dynamic minus uniform. Lower costs are
better; success higher is better. Exact-zero ties are descriptive, not a statistical
equivalence claim. Summaries use eligible-root denominators only, never eight arms
as eight independent roots. No p-values or population/held-out efficacy inference.

1176 upstream predictions are charged once, 8 physical branches retain their
actual attempts/returns, and endpoint encodes stay separate. Four shared inherited
root Jacobians retain unknown acquisition costs (null, not zero), and total pipeline
equivalence remains false. No setup/reset/warmup/collection costs are inferred.
All four roots remain opened calibration; new independent roots=0; P1/P3=false.

## Reproduce the arithmetic audit

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_p3_action_jacobian_outcome_audit_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p3_action_jacobian_outcome_audit_20261027_v1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_outcome_audit_20261027_v1
```

Output is exclusive NEW and includes report.json plus an independent audit source
seal. Tests include census/root/order mutations, false full-path promotion,
post-guard dispatch, counter/label/encoding corruption and unknown-cost relabeling.
The experiment-code skill informed evidence/resource separation; TDD established
failing import before implementation and guards comparison/negative regressions.
