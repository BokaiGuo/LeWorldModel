# Depth-3 matched root-candidate source development protocol

Purpose: test whether the root candidate source helps a common depth-3 Atlas
search on the eight already exposed exact-unicycle goals. This is not a
learned-model, physical-control, independent-confirmation or gate experiment.

Fix goals 0–7 of the existing 64-goal angular panel and seeds 42, 43, 44.
All arms run unchanged Atlas `search`, depth 3 and beam width 2. Every root
bank contains zero and the same local progress action, plus exactly two
candidate-source-specific actions: signed first nonzero spectral direction
with the existing 0.5 prefix radius; two uniform random bounded actions; or
opposite maximum turns combined with the shared progress velocity.

Every subsequent node in all arms uses exactly the same deterministic bank:
zero, locally recomputed progress, and opposite maximum turns at the local
progress velocity. Thus the suffix search, objective, depth, pruning and
selection rule are shared. There is no feedback in this exact-dynamics
development experiment. Each trajectory is verified by Atlas's direct versus
split replay check. Do not inject externally optimized actions.

Compute the same local progress geometry at every visited node in each arm;
this intentionally charges the random and ordinary arms for unused geometry.
Require equality of measured backend rollout/Jacobian calls and action steps,
but record end-to-end wall time separately. Padding unused geometry does not
establish equal total compute, and seeds on deterministic arms are not
independent tasks. Fixed-turn uses task-specific action coordinates; report
it as an ordinary candidate source, not a general-purpose strong baseline.

Record all root banks, selected complete programs, costs, counts, seed,
source hashes and duration. Primary descriptive metric: terminal Euclidean
position error; give root-level results and paired wins/ties/losses. No
reconfiguration-mechanism, visual or admission claim follows. Preserve all
previous results and do not tune this protocol on these outcomes.
