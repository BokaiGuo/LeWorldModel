# v11 same-bank physical score diagnostic

The predeclared first 64 complete chains from the sealed first v11 decision
were replayed at the identical native physical boundary. All 64 completed.
The model predictor scores were saved before physical replay. A separately
isolated process used the frozen encoder and the same preprocessing to score
the actual terminal RGBs against the goal RGB; it received images only.

| Comparison, lower is better | Agree / valid non-tied pairs | Tied pairs |
|---|---:|---:|
| Predicted MSE vs actual-terminal encoded MSE | 1617 / 1920 (84.2%) | 96 |
| Predicted MSE vs physical joint XY error | 1736 / 2002 (86.7%) | 14 |
| Actual-terminal encoded MSE vs physical joint XY error | 1503 / 1920 (78.3%) | 96 |
| Predicted MSE vs physical object XY error | 0 / 63 (0%) | 1953 |
| Actual-terminal encoded MSE vs physical object XY error | 0 / 63 (0%) | 1953 |

Although the bank has 64 distinct action chains, the object-position metric
has only **two distinct values**. All 63 non-tied object comparisons involve
the same exceptional outcome; the 0% figures therefore do not establish a
general inverse relationship. The goal was constructed from a witness and
the candidates are correlated. Pairwise comparisons are not independent
samples, and these results do not certify task-wide scoring or predict
performance on the stopped random goal.

A separate-process audit rebuilt the bank from the original records, checked
the image-only encoder input and hashes, replayed all 64 chains with the same
native backend, and independently recomputed metrics and pair ordering. It
did not independently rerun the neural encoder or use another physics
implementation. See [raw records and audit](../le-wm-main/results/pc_jepa_v11_same_bank_physical_score_diag_20261104_codex/).
No PC gate changes status. The remaining scoring question requires a separately
specified candidate bank with actual variation in contact and object pose,
not further interpretation of these 63 related comparisons.
