# Matched-call exact-unicycle development result

The frozen development protocol ran on the same eight previously exposed
goals, with seeds 42, 43 and 44. Mean terminal position error over 24
goal/seed combinations was:

| Extra prefix candidates | Mean error | Extra candidate selected |
| --- | ---: | ---: |
| Two uniform random actions | 0.304264 | 8 / 24 |
| Both signs of first task-weighted spectral direction | 0.356722 | 6 / 24 |
| Two fixed opposite-turn actions at shared progress velocity | 0.009901 | 24 / 24 |

The ordinary arm had lower error in all 24 paired goal/seed comparisons
against each other arm. The repeated seeds do not create independent goals:
only the random arm changes across seeds. In this unicycle, the zero-action
position Jacobian has a turn nullspace; the nonzero task-weighted spectral
direction alone does not propose the turn candidates. The ordinary arm uses
the task's action coordinates and fixed maximum turn, which is a meaningful
design advantage and is not an independently tuned general-purpose baseline.

Each arm per goal/seed used 28 attempted and returned backend rollouts
(32 transition steps) and five attempted and returned analytic Jacobian
calls (five differentiated action steps). Across 72 runs, that is 2,016
rollout attempts and returns, 2,304 rollout action steps, and 360 Jacobian
attempts and returns over 360 action steps. The replay calls are included.
Test execution incurs additional calls and is not included in these experiment
totals. No model encodings, neural forwards/backwards, environment steps or
physical rollouts were executed. Calls and action steps match; runtime,
allocation, candidate-generation arithmetic and general compute do not.

The result compares two extra candidates inside a fixed four-prefix,
single-suffix Atlas-style ablation. It is not the full v2 beam search and is
not independent confirmation. It cannot establish mechanism, real-world
performance, generalization, or any scientific gate. The full v2 smoke had
76 rollout and 7 Jacobian calls per root and a different search; its reported
0.141953 mean error is not a matched comparison to these results.

The new tests plus existing v2 search and injection tests: **10 passed**.
The result JSON contains all candidate actions, costs, direct witness hashes,
per-run counts, configuration, source hashes and NumPy version.

- Protocol: `docs/PC_ATLAS_MATCHED_EXACT_DEV_20261103_EXCLUSIVE.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_atlas_matched_exact_dev_20261103_exclusive.py`
- Tests: `le-wm-main/tests/test_pc_atlas_matched_exact_dev_20261103_exclusive.py`
- Data: `le-wm-main/results/pc_atlas_matched_exact_dev_20261103_exclusive.json`
