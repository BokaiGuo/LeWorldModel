# v13 isolated synthetic-observation smoke: result

All six predeclared cases completed: v13 and matched global-bank CEM at
15-, 10-, and 5-action horizons. Each used the real frozen JEPA model with
synthetic RGB, proprioception, action history and carry; no physical state,
task observations, simulator, or action dispatch were used.

The saved-data audit passed all six cases. Each consumed 1920 prediction
tokens; the common seeded global bank had 65 chains initially and 66 when a
synthetic carry was supplied. Per-step inputs were identical across arms.
The selected complete chain was scored, within the bounded action domain,
and no worse than the scored full-chain incumbent within `1e-5`. A separate
saved-data pair audit verified the zero-first-packet comparator, pair scores,
bank choice, selected chain and carry inclusion. The second audit was added
after execution, and only inspects already saved outputs.

The six workers ran under the original network/filesystem isolation using a
host execution environment that permits `bwrap`; the previously stopped
contact-root experiment was not rerun. At the 10-step synthetic case,
planning took about 17.28 seconds per arm on this GPU, with 497 predictor
module calls and 10 SVDs for v13 versus 482 predictor module calls and no
response SVD for CEM. These are illustrative engineering timings only,
not a validated quality–cost curve.

Synthetic scores do not measure control quality. The diagnostics do not
verify physical carry-over, real observation-history binding, closed-loop
execution, independently reproduced physics, source independence, or any
PC gate. The fixed six-attempt smoke protocol is complete; no task
performance conclusion is drawn.

Protocol: `PC_JEPA_V13_SYNTHETIC_ISOLATION_SMOKE_PROTOCOL_20261104_CODEX.md`.
Inputs, output evaluations, logs and audits:
`le-wm-main/results/pc_jepa_v13_synthetic_isolation_smoke_20261104_codex/`.
