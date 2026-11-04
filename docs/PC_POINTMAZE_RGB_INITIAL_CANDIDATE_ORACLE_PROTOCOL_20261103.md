# RGB initial CEM candidate physical decomposition: development protocol

Use all previously sealed proposals and prediction scores from the initial
RGB-position random-CEM and spectral-initialized-CEM branches of
`a1_pointmaze_rgb_state_full_planning_20261103_DEV01`, roots 1884--1887.
For every root and branch, replay all 3 x 16 two-packet proposals from the
same reset state with the audited position backend, using each packet as
five high-level actions. Measure physical endpoint distance to the same
archive frame-10 goal, wrapper-style endpoint success (< 0.5), model score,
the actual CEM-selected candidate, per-bank physical oracle, and for spectral
branches the four injected first-generation proposal outcomes. Verify saved
CEM proposal-generation and selected-plan trace against the sealed noise.

This is a post-outcome diagnostic: privileged goal/reset and physical oracle
are evaluator-only and never used for action selection. The replay executes
the entire two-packet open-loop candidate; it does not represent the prior
cp5-replanned closed-loop trajectory. The adaptive banks have shared
innovations and overlap, were generated on four exposed development roots,
and spectral adds JVP/SVD overhead. Counts are not independent roots, model
scores should not be compared across different contexts, and neither an
oracle nor equal prediction count implies an equal-budget strong-baseline
advantage or P1. Save all candidates, predicted and physical outcomes,
selection identities, backend receipt and SHA bindings.
