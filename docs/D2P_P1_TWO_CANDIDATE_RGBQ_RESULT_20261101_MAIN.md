# Same-cp5 two-candidate endpoint diagnostic

On the already observed development root, the exact same actual cp5 context,
goal, normalizer and checkpoint were used to recompute the two sealed suffix
predictions and encode their separately executed actual cp20 RGB frames.
The two fixed action sequences were not selected based on this diagnostic.

| Suffix | Predicted endpoint Q | Actual RGB endpoint Q | Physical jointXY | Native success |
|---|---:|---:|---:|---|
| structured own | 0.12811 | 0.10009 | 120.4565 | no |
| CEM | 0.33549 | 0.46264 | 345.9236 | no |

For this one pair, all three orderings favor structured own. This is not a
general ranking-accuracy estimate: the pair and development root were already
observed, physical outcomes are correlated, and neither reaches native success.
It does not establish independent P1 mechanism or full-controller superiority.
Source binding and frozen-model hash checks passed. Six predictor and six
action-encoder calls and two actual RGB encodes were charged for this
diagnostic; underlying search and physical collection costs are separate.
The action sequences were not physically re-executed here.

Report:
`le-wm-main/results/p1_cp5_two_candidate_rgbq_20261101_leibniz_RUN_MAIN01/report.json`;
SHA-256 `b0469a66287b89740c4d2ceecd48dd575cc0dfef82854f89bcde4487d802943d`.
