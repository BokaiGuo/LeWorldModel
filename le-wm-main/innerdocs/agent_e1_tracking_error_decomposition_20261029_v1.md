# E1 same-time saved tracking / task-goal / native decomposition

This saved-only analysis binds the actual RGB prepare/report/completion/MAIN full receipt and the original physical report/full receipt, retaining all 6×cp0/5/10 positions and all 18 same-root same-time arm pairs. It imports MAIN authentication, not fullstrict or independent neural/gradient verification. It hashes and loads the sealed actual f32 latent arrays and rechecks each original plan, target, raw10, checkpoint, receipt scalar, task and common-goal identity. The original whole replay is the prediction; segmented replay diagnostics are retained independently. Input f32 values are promoted to f64, square products are reduced via `math.fsum`/192; algebraic roundoff is explicitly retained without tolerance. Native state7 metrics are independently rebuilt by the pinned existing E1 physical-summary arithmetic, including original native angle/success and wrapped angle, velocity and XY separately.

For each reference (own target or original common task goal), model MSE = actual MSE + prediction residual energy + signed cross term, with reconstructed value and roundoff. Each target-goal geometric separation and signed target/goal tracking cross term is separately reported. Different target tracking Q is not a cross-arm ranking; the pair table only contrasts common task goal and native metrics, with cross-target tracking delta always null. Missing images remain null; no re-encoding, filling, search, selection or threshold. Two known roots are not fresh confirmation. FLOPs unknown, scientific gate unchanged, all inherited RGB/physical/controller resource records carried independently.

The original RGB consumer uses Python binary64 `**2`, while the reused decomposition kernel uses binary64 array multiplication; each source scalar is independently rebuilt using the original exact formula and strictly compared to the receipt. The decomposition uses its own explicit multiply and saves the original-minus-decomposition tracking-Q difference, with no equality tolerance. In the actual saved inputs, slot 5 cp10 model tracking differs by 2.220446049250313e-16 between those arithmetic routes; this is arithmetic rounding, not an encoder or model discrepancy.

MAIN saved-only commands from repo root (exclusive new path; verify uses actual returned SHA):

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_tracking_error_decomposition_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_tracking_error_decomposition_20261029_v1 build --output results/agent_e1_tracking_error_decomposition_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_tracking_error_decomposition_20261029_v1 verify --output results/agent_e1_tracking_error_decomposition_20261029_v1_MAIN01.json --sha256 ACTUAL_SHA
```
