# Fixed-panel C3 saved receipt analysis

This preregisters only engineering/accounting outcomes before reading MAIN01: all
12 ordered known-root × K1/K17 × fixedH1/staticH3-progress/adaptive slots,
statuses, available complete plans, R group and eligible counts, all 27 cap,
attempt and completed axes, native calls, AD primal and Jacobian element counts,
ledger digest and inclusive wall observations. No rank, winner, outcome-based
filter, aggregate across K, or partial plan promotion. Two roots are not fresh.

Each K is a shared 27-axis cap envelope, not equal actual spend, wall or FLOPs.
Inclusive nested wall figures cannot be added. Scalar backward, CUDA time and
FLOPs remain unknown, not zero. Q at different horizons is not a common physical
endpoint. This is not proof of P1, full C3, physical benefit or original R
equivalence. Status/refusals/no-witness retain their full denominators.

Input is the actual SHA-pinned HOST strict receipt, not a RAM receipt; the
consumer validates receipt identity and the fixed producer schema and inherits
MAIN's full saved reconstruction. It does not independently repeat that strict
verification, certify an arbitrary self-created receipt, read model artifacts,
or perform neural or physical computation. MAIN must supply the authentic
receipt SHA from its strict run. The producer triple and protocol are pinned.

From repository root, after MAIN's receipt exists:

```
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_preregistered_analysis_20261029_v1 analyze --receipt RECEIPT --receipt-sha RECEIPT_SHA --analysis results/agent_p3_c3_preregistered_analysis_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_preregistered_analysis_20261029_v1 verify --receipt RECEIPT --receipt-sha RECEIPT_SHA --analysis results/agent_p3_c3_preregistered_analysis_20261029_v1_MAIN01.json --analysis-sha ANALYSIS_SHA
.venv/bin/python -m pytest -q tests/test_agent_p3_c3_preregistered_analysis_20261029_v1.py
```

`analyze` creates a new exclusive file; `verify` reconstructs the entire analysis.
Neither invokes the producer's full consume. Failure or incomplete receipt is
rejected, not replaced by zeros or selected remaining slots.
