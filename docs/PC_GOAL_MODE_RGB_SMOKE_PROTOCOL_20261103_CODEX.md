# Goal-conditioned compiler RGB integration smoke

Development wiring test only: replay development root 0 with the existing
PointMaze shadow RGB bridge, its frozen checkpoint, true warmup RGB/action
history and RGB goal. Run goal-conditioned mode ranking at the post-warmup
root. Use the existing model-derived projected position coordinates for both
response and goal. Predict five-step prefixes, optimize ten-step suffixes
with the existing CEM and choose by the model's full fifteen-step score.
Save all generated candidates and the selected action chain before executing
that chain in the same RGB backend. No physical outcome is supplied to the
planner. Replayed development data are not independent.

Use seed 2026110700 and 12 CEM samples per retained prefix. A nondegenerate
three-candidate run uses 21 one-token response samples, three one-token
prefix scores, and 36 three-token complete-chain samples: 132 token samples.
Set a 132-token cap; fewer candidates may use less. SVD and other operations
are recorded separately, so token accounting is not equal total compute.
Model encoding is recorded separately and shared setup is not fully metered.
The zero prefix is a candidate, not a formal abstention mechanism.

Output hashes bind the protocol, code, checkpoint and previous history. This
smoke test does not establish PC2, PC3, unbiased performance, full cost
certification, or a new non-immediate mechanism claim.
