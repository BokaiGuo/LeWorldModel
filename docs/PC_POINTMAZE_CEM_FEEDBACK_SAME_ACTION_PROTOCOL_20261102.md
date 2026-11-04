# Fixed CEM candidates: feedback and score-binding diagnostic

Development protocol written before running this diagnostic. Use all 384
initial two-packet candidates from the sealed all-candidate replay
`a0_pointmaze_cem_all_candidates_20261102_DEV01.json`, SHA-256
`7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb`,
and its source CEM report. Preserve root, candidate, and action order.
Replay each first packet to acquire real step-5 RGB; use the already sealed
same-action step-10 RGB. Compare (1) open-loop two-packet model prediction,
(2) a one-packet prediction after real step-5 RGB refresh, and (3) actual
step-10 RGB encoding. Use the original initial RGB task mask and goal for
all three scoring paths, so only the prediction context differs. Also
measure task-mask prediction-to-actual feature MSE. Do not train anything or
use real terminal outcomes to select actions.

Recalculate all original predicted task scores and compare against the CEM
source; report maximum absolute error and selected-index agreement. Check
source, image arrays, checkpoint, dataset and model hashes; check every
regenerated reset and selected-prefix image. Report within-root/arm pair
reversals and ties and per-candidate prediction errors for each model path.
The root and arm are the relevant clusters; 9,024 correlated pairs are not
independent samples. A post-cp5 score is a diagnostic of feedback prediction,
not a new physical closed-loop outcome, and cannot establish P1, model
training-set independence, or complete interface correctness.
