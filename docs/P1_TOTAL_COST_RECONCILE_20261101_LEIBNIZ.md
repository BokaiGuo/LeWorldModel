# P1 known-root total-cost reconciliation

This is a retrospective, read-only reconciliation of three pinned reports; it
is not an executable total-budget guard, full-cost certification, equal-compute
claim, or P1 admission. The exclusive script verifies each prepared source
manifest, report and physical journal hash, operation attempt/return pairs,
terminal counts, fixed node roster and the existing per-node ledger contracts
and identity. It reports environment/visual counts separately from the node
reservations, fixing the accounting presentation gap where node encode/physics
zeros could be misread as whole-run zeros. Nested public/physics operations and
selection/replay/prediction overlap: never sum these axes as independent work.

Run from `le-wm-main` with a new unused output name:

```sh
.venv/bin/python -m experiments.spectral_reachability.p1_total_cost_reconcile_20261101_leibniz --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_total_cost_reconcile_20261101_leibniz_RUN02.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_total_cost_reconcile_20261101_leibniz.py
```

The pinned RUN01 receipt has SHA256
`e1bae3f6adced3cae266bbb38e36578fc2d863b78f39b13dce5dd481e84234f8`.
All source manifests matched at execution. The independently checked native
root0 action/RGB binding is in the earlier
`P1_COMMON_CP10_PAIR_AUDIT_20261101_LEIBNIZ.md`; this reconciliation alone
does not independently recheck the selected action or all RGB bytes.

| Known root / arm | Actual control | RGB encodes | Physics substeps | Node predict reserved | Replay/selection reserved | Backward / VJP |
|---|---:|---:|---:|---:|---:|---:|
| root0 structured | 13 | 3 | 234 | 252 | 42 / 42 | 624 / 624 |
| root0 CEM | 17 | 4 | 274 | 775 | 548 / 548 | 0 / 0 |
| root1 CEM | 6 | 2 | 164 | 389 | 162 / 162 | 0 / 0 |

These are self-consistent logged operations and charged upper bounds, not
independent coverage measurements. Counts include setup and warmup physics;
they are not just the control steps. Root1 CEM stopped at six controls and
nodes 10 and 15 were not started. No root1 structured common-budget report is
pinned here, so the missing arm is `null`, never a zero-cost result. Root0
pair target/model identity is checked; root1 is a different root and target,
not a cross-root method comparison.

Important remaining gap: per-node caps do not enforce a single precommitted
whole-run cost vector. The runtime physical/visual meter is created inside
`execute`, after model loading; offline cp10 goal construction/encoding,
source preparation, model load, and independent coverage/GPU compute are not
fully metered or allocated. Their cost remains unknown, not zero. A real fix
for future prospective comparisons needs a separately reviewed exclusive host
that binds both arms and root/goal before execution, charges attributable
setup/goal and actual RGB/physical dispatch plus planner events against one
precommitted vector, preserves attempt/return on failure, and independently
validates coverage and units; refuse certification if any component is
unbound or unknown. This receipt does not implement that host or claim such
coverage, and does not alter the sealed runners or gates.
