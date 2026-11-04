# v17 E0 multistart baseline: development result

The eight prespecified tasks and all three arms completed. The success
threshold was terminal distance below 0.05.

| Arm | Success | Prediction tokens per task |
| --- | ---: | ---: |
| v13 | 1/8 | 4,095 |
| v17 | 1/8 | 4,095 |
| Four-start CEM | 2/8 | 4,095 |

Both compilers and CEM succeeded on root 2; CEM alone succeeded on root 4.
This panel does not establish a v17 planning advantage. It is a small
development comparison, not PC3 confirmation or an upper bound on CEM.
Broader task provenance is unknown, and equal prediction-token budgets
do not imply equal total cost. Each compiler recorded 18 SVD calls per
task; CEM recorded none.

The runner's `wall_seconds` is sampled after writing the compressed trace,
so it includes serialization as well as planning. It must not be interpreted
as planning-only runtime or used to claim a hardware cost advantage.

The original saved-trace audit checked the common bank, action bounds,
all recorded scores and endpoints using alternate same-environment
geometry, complete-chain incumbents, and prediction-token accounting.
An additional saved-trace audit checks each CEM start is selected from the
stable bank ranking, the four allocations are balanced, every generation's
incumbent is retained, and the final pair selects the recorded minimum.
This is a consistency check, not independent replication or evidence that
the optimizer is fully tuned.

The eight-task stopping rule has been reached; do not rerun, replace roots,
or tune this panel. Protocol:
`PC_V17_E0_MULTISTART_BASELINE_DEV_PROTOCOL_20261104_CODEX.md`.
Raw traces, plan records and both audits:
`le-wm-main/results/pc_v17_e0_multistart_baseline_dev_20261104_codex/`.
