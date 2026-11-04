# PointMaze RGB feedback CEM: paired integration development protocol

This protocol is a follow-up to an already observed single-root goal-mode
smoke, so it is exploratory rather than prospective confirmation. Use the
same real warmup, root, goal, frozen model, action normalization, clock,
three feedback decisions, 15-step model planning horizon and five actions
executed per decision. At each decision the baseline runs direct full-horizon
CEM with 44 complete-chain samples: 132 five-action prediction tokens,
then executes its first five actions and refreshes real RGB. Use seeds
2026110710 + decision index, pre-outcome decision sealing and separate files.

The prior goal-mode arm spends extra encoding/derivative/SVD/selection work
not captured by the token budget. This matched-token integration comparison
does not certify equal total resources or establish an advantage. Record
timing and model operations separately; physical replay is a distinct
post-seal step. Never interpret this reused root as independent confirmation.
