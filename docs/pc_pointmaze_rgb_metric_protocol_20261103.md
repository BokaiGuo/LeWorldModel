# PointMaze fixed-bank RGB metric: retrospective development protocol

Frozen before this implementation's scoring run, but after these banks and
related analyses were exposed. This is not a prospective preregistration.
Use the 624 saved terminal images in
`a0_pointmaze_failure_real_rgb_metric_20261103_DEV01.npz`, the corresponding
saved goal RGBs in `a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.npz`,
and exactly the 12 ordered groups listed in the two associated receipts.
Bind all source receipts and arrays by SHA-256; check group identity, counts,
and the prior receipt's exact replay-position binding. No new simulation,
training, candidate generation, subset selection, or threshold fitting.

Scoring is a separate first stage: only image arrays and their group
identities enter the scoring function. Reproduce the existing green marker
detector: `g>100`, `g>1.25*r`, `g>1.15*b`, `r<100`; require 4–500 pixels,
span less than 20 pixels in each axis, and centroid at least 14 pixels from
every edge. Score is the Euclidean distance in pixels between the terminal
and goal marker centroids. Both images use the same declared top-down camera,
so pixel distances give the same ranking as the existing common-scale static
projection. Do not use camera offsets, physical state, physical goal, latent
values, or physical outcomes in scoring. Record all detection failures; a
group with any failed detection receives no ranking, with no imputation or
outcome-based exclusion. Persist per-candidate scores before evaluation.

Evaluation stage: load the fixed saved scores, physical distances,
predicted-plan latent costs (`*_scores`), and actual-terminal-RGB latent
costs (`*_real_cost`). Verify source hashes and group/candidate alignment.
Within each bank count pairwise reversed orderings, comparable pairs, and
ties, using exact numerical equality; also report the physical distance
at each score's deterministic first argmin. Report all three comparisons
against physical distance and both latent scores against RGB geometry.
Aggregate counts are descriptive pair-weighted summaries, not independent
trials: banks, candidates and roots are dependent. Retain all rows.

Saved terminal RGB is a real future observation and therefore an offline
oracle, not an implementable pre-action scoring signal. Existing receipt
reports exact endpoint-position equality between the old render replay and
physical oracle; this run does not independently rerun that physics.
Rendering/integration timing, occlusion, quantization, static scene and
camera assumptions limit interpretation. No closed-loop, independent
evidence, causal mechanism, or controller-performance claim is permitted.
