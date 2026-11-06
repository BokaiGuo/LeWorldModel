# v25 successor-probe cost ablation: development protocol

Use exactly eight new exact-Wall tasks with start `(-0.5, ±0.19, ±1.21)`
and goal `(0.5, ±0.18)`. Check exact overlap with enumerated v25 and prior
panels; broader provenance is unknown. Do not replace or revisit roots.

Each paired root uses the same nominal actions, action bounds, target,
global proposal seed, optimizer and 1,920 prediction-token cap. Compare:

1. Frozen v25 response with successor probes and response retention.
2. A checked v25 code-equivalent ablation disabling successor probes:
   retain using v25's no-probe diversity fallback and reallocate the
   unused token allowance to the same iterative full-chain optimizer.

The ablation retains the earlier local response computations for generating
spectral candidates. It therefore isolates the *combined effect* of
successor probes and their response retention, including changed allocation
and all downstream search consequences; it is not a no-response-at-all
ablation. Enabled ablation versus frozen v25 must first produce exactly
identical prediction calls and chosen actions in a deterministic regression
test. Both arms count all predictor calls and preserve complete-chain
incumbents, paired abstention and failure denominators.

Persist every prediction attempt, returned result or failure using an
external journal. Report per-root distance, success under the existing
`<0.05` threshold, paired differences, actual prediction tokens,
full-horizon sample counts, successor probe counts, number of refinement
samples and SVD calls. Prediction tokens are the primary matched axis;
wall time and non-prediction costs are separately reported and do not
establish matched GPU, visual encoding or feedback costs. Audit journal
hashes, same initial global bank, actual selected actions and scores, and
alternate same-environment geometry. This is a development experiment,
not PC2 or PC3 confirmation; negative results remain valid.
