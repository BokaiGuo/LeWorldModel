# Fixed-prefix saved gradient and shadow candidate diagnostic

This saved-only exploratory diagnostic preserves all 64 groups, 576 original
candidate positions and eight paid suffix gradient steps per group. It reports
each stage Q relative to the original initial Q, each group's best terminal Q
relative to both the source suffix and the original winner, and all eight
same-iteration terminal changes. No physical outcome, root filtering, training
or altered original gate is used.

For each saved step it checks that the gradient forward's entire context is
the preceding candidate's context, recomputes `g_logit=g_raw*(1-tanh(x)^2)`
using the actual float32 chart, and exactly reconstructs the original eta=1
projection, next logits and raw bits. It reports the inner product of that
saved chart gradient with the actual projected displacement and the actual
terminal-Q difference. Their difference is a finite-step linearization
residual, **not** proof of Hessian curvature, derivative correctness or a bug.
The host strict separately reconstructs native IO and scalar objective, but
does not establish numerical gradient correctness; finite differences require
separately budgeted new model calls and explicit authorization.

At each of the 512 paid gradient points a deterministic shadow proposal is
`project(x - 0.5*g_logit)` in the original ball/chart. It is a fixed arithmetic
candidate, not a new optimization trajectory: it starts from each existing
eta=1 trajectory point and has **no evaluated score**, no witness status and
no physical plan. The 512 shadow candidates have a separate denominator, zero
new model calls and zero available scores. They cannot establish improvement
or substitute for the original 576 candidates. Any future score or experiment
requires a separately frozen common budget and no outcome-based root selection.

`analyze()` is a pure synthetic-testable join and assumes full strict has
already been established. `build()` invokes the fixed-prefix handoff full
authority rebuild, including inherited host and source verification; no saved
receipt status alone substitutes for that rebuild. This may read sealed old
physical source artifacts for lineage, but no physical outcome is used to
select a candidate. If even lineage reads are prohibited, do not invoke build.
The worker ran only RAM fixtures, not production analysis, full strict, CUDA,
or physics. P1, freshness and physical release remain false.

From the repository root, tests:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1.py
```

After MAIN review, actual paths and hashes must be supplied:

```
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1 analyze --host-output /ABS/HOST_OUTPUT --host-report-sha REPORT_SHA --host-prepared /ABS/HOST_PREPARE --host-prepare-sha PREPARE_SHA --host-receipt /ABS/HOST_RECEIPT --host-receipt-sha RECEIPT_SHA --output /ABS/results/agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1 verify --analysis /ABS/results/agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA
```
