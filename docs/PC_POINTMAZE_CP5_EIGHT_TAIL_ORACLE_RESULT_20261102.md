# Fixed CP5 eight-tail physical oracle decomposition

The fixed development protocol executed all 128 slots: four exposed roots,
four sealed prefixes per root, and eight fixed tails per prefix. Every
candidate produced a native step-10 RGB and was encoded on host CUDA. All
128 step-5 images matched their source prefix; all 16 source-selected step-10
images and offline position-backend endpoints reproduced the source.

For each prefix, a privileged physical oracle picks the lowest position
distance among its eight tails. The table compares other selectors' mean
distance regret relative to that same candidate-limited oracle:

| Tail selector | Mean position-distance regret | Same index as physical oracle |
| --- | ---: | ---: |
| Single-frame model prediction | 0.17644 | 1/16 |
| Real-history model prediction | 0.21390 | 0/16 |
| Actual successor RGB latent goal cost | 0.08076 | 7/16 |
| Physical candidate oracle | 0 | 16/16 |

The actual-successor and physical oracles use future outcome information
and cannot be deployed as planners. Their comparisons show that the existing
bank contains lower-distance alternatives on 15 of the 16 source-selected
prefixes; they do not prove this bank can solve the full task, nor do they
identify a fair-budget algorithm that can find those alternatives.

Across the 16 sets, each with 28 within-set candidate pairs, exact-sign
ranking comparisons gave:

| Comparison | Reversed pairs | Exact ties |
| --- | ---: | ---: |
| Single-frame predicted latent cost vs actual RGB latent cost | 259/448 | 0 |
| History predicted latent cost vs actual RGB latent cost | 267/448 | 0 |
| Actual RGB latent cost vs offline position distance | 160/448 | 0 |

There are only four development roots. Prefixes and candidates share starts,
some action chains recur, and these pairs are **not independent samples**.
Exact equality defines ties; no numerical-margin threshold was predeclared.
These results demonstrate both prediction-to-encoding and
encoding-to-physical ranking discrepancies on this panel, but not their
exclusive cause. More accurate predictions would not necessarily fix a
misaligned task metric, and replacing a model prediction with a real
successor is a privileged diagnostic rather than a proposed deployment.

All endpoint positions are from the separate offline evaluator, not directly
captured state in the native RGB bridge. The cross-backend replay check was
previously completed for the 16 source-selected chains only. The model uses
constant raw-zero proprio, the action normalizer uses the full archive, the
camera was calibrated on an exposed root, and checkpoint training membership
is unknown. Position distance is not native task success. The study is not
independent confirmation, perfect-dynamics MPC, equal-budget performance or
P1 mechanism evidence.

The NPZ archive stores each candidate RGB, pooled latent, physical endpoint
and goal latent; the JSON stores all costs, hashes, selection indices, regrets
and pair counts. Five targeted tests passed, including recomputation from
saved arrays and the previous history/physical bindings; they did not rerun
neural encoding or independently authenticate every physical endpoint.

- Runner: `le-wm-main/experiments/a1_pointmaze_cp5_eight_tail_oracle_20261102.py`
- Protocol: `docs/PC_POINTMAZE_CP5_EIGHT_TAIL_ORACLE_PROTOCOL_20261102.md`
- Result: `le-wm-main/results/a1_pointmaze_cp5_eight_tail_oracle_20261102_DEV01.json`
- Result SHA-256:
  `8e8b3a938da82829c722b646b09dfd0dfdb395974960a81d83445c7fb3fec129`
- Tests: `le-wm-main/tests/test_a1_pointmaze_cp5_eight_tail_oracle_20261102.py`
