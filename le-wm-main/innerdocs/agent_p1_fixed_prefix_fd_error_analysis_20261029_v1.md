# Fixed-prefix FD saved error analysis

This saved-only analyzer requires actual external FD host output/report SHA and
persisted receipt path/SHA. It rebuilds the complete frozen host consume and
requires canonical byte equality with the persisted receipt before output;
partial receipt, missing group, wrong source, mutation or missing/null mismatch
fails closed. Full consume also reconstructs source lineage and may read old
physical lineage artifacts, but physical outcomes never choose analysis rows.
It does not run a model or physical environment.

All 64 groups and 2560 pairs are retained. Each raw/chart × scale cell has
denominator 640. A valid row reports signed and absolute `FD - saved_AD`,
`abs(saved_AD)` as the explicit relative denominator, and absolute relative
error. AD exactly zero makes the relative error null with a reason, never an
epsilon. Signs are negative/zero/positive; relation distinguishes opposite,
same nonzero, one zero and both zero. Invalid pair metrics remain null with
all original causes. Two-scale comparison is matched by group/space/coordinate,
with `small minus large` FD difference only when both exist; each space has
640 cross-scale slots. Summaries are descriptive means/max/counts with full
denominators, not inference, p-values or pass thresholds.

The FD objective is endpoint float64 MSE while the saved native AD is of a
float32 loss; finite-step, float32 quantization, reduction and boundary effects
remain. These fixed two known roots, coordinates and two scales cannot prove
global derivative correctness or P1. No cutoff, score-based exclusion, new
action, fresh-root claim or physical release is introduced.

The worker runs synthetic RAM/temporary-file tests only; actual receipt/full
strict is MAIN's task. Pure `analyze()` assumes a fully authenticated input;
only `build()` establishes it. No production neural/physics work occurs here.

From project root:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_fd_error_analysis_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_error_analysis_20261029_v1 analyze --output /ABS/FD_OUTPUT --report-sha REPORT_SHA --receipt /ABS/FD_RECEIPT --receipt-sha RECEIPT_SHA --save /ABS/results/agent_p1_fixed_prefix_fd_error_analysis_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_error_analysis_20261029_v1 verify --analysis /ABS/results/agent_p1_fixed_prefix_fd_error_analysis_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA
```
