# Shared-pool endpoint pusher confounding diagnostic

Use all four exposed shared-prefix pools and their already sealed endpoint
embeddings, scores and physical replays. For each task compare all unordered
candidate pairs whose final object centers differ by at most five simulator
pixels and whose wrapped object angles differ by at most 0.1 radians. Retain
pairs where absolute difference in pusher distance to the target-frame pusher
position is at least 20 pixels. For each task and each of actual-encoding,
single-frame predicted, and three-frame predicted cost, report how often the
candidate with the closer pusher has lower cost, separately counting ties.
Report qualifying pair counts and physical success disagreements; include
all tasks even when there are no successful candidates.

Also report task 1's single final-success candidate's pusher distance and its
number of qualifying matches, without using it to define the candidate pool
or thresholds. The thresholds are developer diagnostic settings set before
running this analysis, not evidence of exact object-pose equivalence.
Simulator state is evaluation-only. Pairs share candidates and are not
independent samples; pusher distance may be confounded by residual object
pose, occlusion, background, and representation effects. This is an
observational hypothesis check, not a causal intervention, deployable score,
or confirmation experiment.
