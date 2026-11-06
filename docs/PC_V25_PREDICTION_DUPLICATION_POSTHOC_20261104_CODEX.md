# v25 repeated predictions: post-hoc E0 check

Within each of the eight sealed same-chain planning contexts, the saved
prediction journal contains 38 exactly repeated action samples, totalling
102 five-step prediction tokens across 19 batches with at least one
duplicate. All repeated actions had identical saved endpoint features and
scores. Equality is byte-exact for float32 action sequences, including
horizon; this does not merge approximately equal actions or contexts.

This identifies potential avoidable prediction work, not a demonstrated
correctness bug or a proven speedup. The 102-token figure is a
retrospective upper bound for exact reuse on these fixed recorded
trajectories, not the cost or outcome of a new planner. A cache-enabled
planner can change later sampling and token allocation, and must count
cache lookup and storage costs. A valid implementation must bind cache
entries to the complete model/context/goal and action contract, prevent
reuse after RGB observation or model changes, and distinguish proposed
from actually evaluated samples. The internal v25 budget currently counts
submitted samples, so wrapping its predictor alone would not reallocate
the savings; the budget contract must explicitly change in a new version.
Any complete-method comparison must apply equivalent caching permission
and accounting to baseline arms.

This result neither certifies visual predictor determinism nor implies a
PC2/PC3 pass. Raw same-context duplicate audit:
`le-wm-main/results/pc_v25_e0_prospective_same_chain_dev_20261104_codex/prediction_duplicate_posthoc.json`.
