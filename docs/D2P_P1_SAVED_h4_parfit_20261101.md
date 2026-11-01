# H4 saved-result checking

The standalone checker reads an existing run without writing into it:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m experiments.spectral_reachability.p1_saved_h4_parfit_20261101 \
  --directory /absolute/path/to/p1_reverse_h4_parfit_20261101_ARM_RUN_TAG \
  --report-sha256 REPORT_SHA256
```

Use the SHA returned by the run, or independently calculate the `report.json`
SHA. Strict mode requires one arm with all 20 returned controls and four
sealed nodes. It checks the source closure and prepared receipt, full physical
event hash chain and boundary seals, 20 dispatch/return actions, node trace and
contract hashes, durable ledger consumption, hook charges, H4 exchange and
selection identities, full-plan receipts, and previous-tail/current-RGB
binding. It does not independently rerun the model, recompute Q scores,
replay the simulator, prove GPU kernel counts, or establish P1 admission.

For an early native termination, truncation, or another incomplete run, add
`--status-only`. This separately reports the recorded branch, returned step
count, sealed boundaries, success/termination flags and errors. It checks
only the externally supplied report SHA; it does not validate source closure,
actions, budgets or physical event integrity, and does not promote an early
stop to a full20 result. Native stopping or truncation is not automatically an
engineering failure, and a genuine failure is not automatically censored data.

The strict checker was also run read-only on structured `RUN_MAIN01`, with
report SHA `2ff3d42d93f2f5ecf34bc675a9f6e219e7a9ab2d4a1c0742c7ce3ed3f9b14bff`:
all four node integrity checks passed. This is a saved-artifact integrity
result, not a scientific replication or performance admission.
