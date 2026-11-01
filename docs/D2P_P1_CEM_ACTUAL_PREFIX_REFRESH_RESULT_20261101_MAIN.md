# CEM same-action actual-prefix prediction diagnostic

The fixed CEM actions and real RGB latents were compared on host CUDA under
`p1_cem_actual_prefix_refresh_20261101_leibniz.md`. Both arms start at the
same cp5 and use identical actual action packets. The open arm propagates
its own predictions; the actual-prefix arm receives the sealed real RGB
latent after each packet. This is a teacher-forced prediction diagnostic, not
a feedback controller and not a new physical execution.

| Physical step | Open latent RMSE | Actual-prefix latent RMSE | Open Q-projected RMSE | Actual-prefix Q-projected RMSE |
|---|---:|---:|---:|---:|
| 10 | 0.64857 | 0.64857 | 0.54430 | 0.54430 |
| 15 | 0.97557 | 0.52201 | 0.80907 | 0.42296 |
| 20 | 1.04910 | 0.39348 | 0.72033 | 0.30902 |

The first packet matches as required. Replacing the imagined prefix with
actual encoded observations improves both measured prediction errors for
the subsequent packets on this one known trajectory. This cannot establish
population-level improvement, a better planning decision, or physical task
improvement. The actual physical jointXY became worse over this same
trajectory, reinforcing that RGB-Q and physical task outcomes remain
separate. No P1 gate was changed.

Frozen-model and transitive-source checks passed; the original open rollout
and terminal Q were reproduced. The new ledger records six predict calls and
six action-encoder calls, no new encoding or physical action; previously
collected RGB encoding, search and physical execution cost are not free.
The ledger is dispatch accounting, not independent hardware metering.
Report:
`le-wm-main/results/p1_cem_actual_prefix_refresh_20261101_leibniz_RUN_MAIN01/report.json`;
SHA-256 `bff047d470e3922e05160fa468af0f88a314594733e84885d3f61f956676a3eb`.
