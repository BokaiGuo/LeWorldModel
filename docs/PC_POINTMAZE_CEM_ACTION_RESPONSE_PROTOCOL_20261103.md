# Fixed CEM candidate action-response decomposition

Exploratory development diagnostic defined before this analysis. Use all
384 previously executed initial two-packet proposals from
`a0_pointmaze_cem_all_candidates_20261102_DEV01.json` with SHA-256
`7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb`,
its sealed images/actions, and original CEM source model and initial task mask.
At each root/arm recompute the frozen host-CUDA model's two-packet visual
predictions and re-encode the same candidates' real cp10 RGB. Require
recomputed source goal costs and actual RGB costs to agree with the saved
report. Same actions, timestamps, mask and goal are used throughout.

Flatten model and actual features at source task-mask tokens. Subtract the
candidate mean separately from each set. Report centered feature cosine,
least-squares gain of real variation on predicted variation, centered
prediction/actual norm ratio, and centered relative squared error; all are
descriptive and computed per root/arm, with a zero-denominator fail-closed
check. Also project centered responses onto the fixed initial-to-goal visual
direction on the same mask and compare signed responses. Mean-centering
uses the complete executed candidate bank and real futures: it is
evaluation-only, not a deployable calibrated planner.

All roots and candidate outcomes are already exposed; candidate slots share
roots and CEM search ancestry. No inference that observations are independent,
no training, no root selection and no P1 claim. Captured RGB and position
backend use different simulation implementations; zero proprio and
full-archive normalizer provenance remain unresolved. Save source hashes,
full per-candidate projected responses and scalar decomposition sufficient
to recalculate the reported quantities.
