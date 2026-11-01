# dev303 four-arm cp5 V13 development run

The four fixed prefixes were followed by separately optimized ten-step tails.
Each planner received its own real cp5 RGB and actual prefix action history,
with the same initialization and nominal per-arm model-call counts. This is a
development-root experiment, not independent confirmation.

- Model preparation SHA-256: `07828b3e7901839389490089b0fac84e583fc259e3a10476262ab0f01e47ef7b`.
- CUDA model report: `le-wm-main/results/p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive_RUN_01/report.json`;
  SHA-256 `3dee19020499deed967cd2d7ba7380c720412f98f7945db699af45f1aca9bdaf`.
- Physical preparation SHA-256: `76a71b0a61e3a6e2e12fb084f7ff64c2a3195f0caacd5139f00781419b7b194f`.
- Physical report: `le-wm-main/results/p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive_RUN_01/report.json`;
  SHA-256 `88fe063efd872ad2e3837489975c841efec19097d818709b08a596d7d20b7507`.
- Saved-chain verifier: all four branches completed the fixed ten-step tail
  with no reported errors; its result remains development-only.

The last recorded transition was not terminated in any arm. The terminal
native rewards were selected -339.0711, zero -359.8180, task_greedy_direct
-291.5349, and random -357.5816. These are descriptive, single-root values,
not an independently verified mechanism or a fair total-budget comparison.
In particular, the common model-call counts exclude uncertified upstream
planning and full resource costs; the outcome does not establish selected
superiority to greedy, independent generalization, or P1 admission.
