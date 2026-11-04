# Fixed-bank decision regret on exposed PointMaze roots

This retrospective diagnostic reuses nine already-executed action chains per
root. The table reports the physical position distance of the candidate
selected by each score, alongside the best distance in that fixed bank.
Lower is better.

| Root | Open prediction | Refreshed prediction | Actual RGB encoding | Bank oracle |
| --- | ---: | ---: | ---: | ---: |
| 8343 | 0.832 | 0.092 | 0.832 | 0.092 |
| 8345 | 1.217 | 0.884 | 0.884 | 0.884 |
| 8349 | 1.294 | 1.100 | 1.100 | 1.100 |
| 8351 | 1.104 | 1.104 | 1.104 | 1.104 |

The refreshed score happens to choose the physical bank oracle in all four
exposed roots, while the actual encoded score chooses a worse candidate on
8343. However, **refreshed scoring here has counterfactual step-five RGB for
all nine separately executed prefixes**. A real planner choosing an initial
prefix cannot access those observations. Actual-encoding and oracle choices
also use future outcomes. This is not closed-loop performance, and choosing
the best available bank action is not necessarily task success. Candidate
chains share roots and were already exposed; no statistical or independent
generalization claim follows.

The result narrows the diagnostic: even a perfect prediction of the encoded
endpoint need not pick the best physical candidate, and branch-specific
refresh can change choices. It does not establish that refreshing an
executed prefix will improve planning. See the
[receipt](../le-wm-main/results/pc_pointmaze_fixed_bank_decision_regret_20261103_CODEX.json)
and [analysis code](../le-wm-main/experiments/pc_pointmaze_fixed_bank_decision_regret_20261103_codex.py).
