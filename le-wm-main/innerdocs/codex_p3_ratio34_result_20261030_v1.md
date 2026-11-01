# P3 fixed rank-allocation exploration result

The earlier patience-5 run used identical per-root bank order across arms.
Root 0 stopped after index 5 because index 0 remained best; root 1 stopped
after index 9 because index 4 remained best. Thus larger maxima generally
did not produce more candidates. In the older non-early-stop journal root 0
index 8 and index 14 had lower model costs than index 0. These are observed
model costs, not evidence of physical improvement.

The new post-observation fixed-horizon protocol completed and saved-consumed
all six host-CUDA slots. Signal weights were [162.32405169996156,
156.2867146609139]; signal/reverse candidate sets had symmetric differences
20 at point 20 and 32 at point 30, independently counted from the journals.
The per-root winner indices (root0, root1) were:

| Point | Uniform | Signal | Reverse |
|---|---|---|---|
| 20 | 8, 4 | 14, 4 | 0, 4 |
| 30 | 16, 4 | 22, 4 | 0, 4 |

The 20-point forward axes each used 48; metric was 24 for uniform, 20 for
signal/reverse. At 30 the respective values were 68, 34 and 30. Signal and
reverse each used Jacobian 2, VJP directions 384, grad API 384, backward
hook 384, SVD 2, scheduler 1; uniform used zero on those five signal axes and
scheduler 1. These axes overlap and must not be summed. Model selection,
signal and scheduling are accounted, but no FLOP or energy equality is
established.

Report SHA-256:
`f2e800d9eaa75b3353837e661bcfe875f3bdb7e65b2d99d09fcee1096e0ade0f`.
No new physical execution was performed. Known roots, observed historical
outcomes and the changed stopping rule preclude independent confirmation or
a clean stopping-rule-controlled comparison to the preceding run. This is
not a scientific gate pass or a claim of P3 advantage.
