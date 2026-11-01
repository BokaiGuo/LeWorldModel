# Root1 saved structured / CEM reconciliation

This independent exclusive receipt adds the subsequently supplied root1
structured report to the prior reconciliation. It does not modify the prior
receipt or source files. The structured prepare followed the visible CEM
outcome (see `P1_COMMON_BUDGET_CEM_ROOT1_PRERUN_20261101_MAIN.md`); this is
not a prospectively preregistered paired comparison.

The consumer checks pinned report, prepare and every listed source hash;
target/root/model, action normalization, projection, common node caps and
schedule; physical and node hash chains, operation attempt/return census and
ledger reservations; structured selection/carry and CEM candidate journals;
and the actual sealed commands, dispatch, per-step RGB pixels/states, goal
and native first-stop calculation. Both arms have exactly the same actual
initial state and RGB pixel SHA
`34017e1e34c6cd40b48f00ffe02f45b07a8079fde237be50ca72815cb6b4f017`.

| Arm | First native stop | Node starts | RGB encode | Physics substeps | Predict reserved | Replay / selection | Backward / VJP | Jacobian / SVD |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| structured | 8 | 0, 5 | 2 | 184 | 168 | 25 / 25 | 416 / 416 | 2 / 2 |
| CEM | 6 | 0, 5 | 2 | 164 | 389 | 162 / 162 | 0 / 0 | 0 / 0 |

Physics includes setup/warmup. Node 10 and 15 were not started; no actions are
padded. These counters have different units and nested scopes and cannot be
added to infer GPU work. The ledger remains `closed_unverified`, and
attributable offline target creation, model loading, independent coverage and
GPU compute remain unknown. Common per-node caps are not a certified common
whole-run budget. Neither two-root evidence nor this receipt demonstrates
fairness, independent confirmation, or P1 admission.

RUN01 receipt SHA256:
`41faf6913dc671cfa1f5172ae41a20c93674af898272e1eee3b8220417f51217`.
To recompute with an unused filename, from `le-wm-main`:

```sh
.venv/bin/python -m experiments.spectral_reachability.p1_total_cost_root1_pair_20261101_leibniz --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_total_cost_root1_pair_20261101_leibniz_RUN02.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_total_cost_root1_pair_20261101_leibniz.py tests/test_p1_total_cost_reconcile_20261101_leibniz.py
```
