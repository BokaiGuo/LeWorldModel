# Response-SVD prefix bank with predicted-successor CEM

This is a post-outcome development experiment on already exposed Reacher roots
20263160 and 20263161, not independent confirmation. Reuse the 13 saved
response-SVD prefixes per root; do not filter them using physical outcomes.
At each predicted successor run the same seeded CEM: two five-action packets,
16 samples, two iterations, four elites. Score with the existing model L2
coordinate mean squared error. The existing selector compares model-predicted
immediate cost and optimized future cost with neutral-parameter prefix, both
using margin 1e-6, and returns null if no candidate qualifies. This selection
rule is exploratory, not the official P1 gate or a physical non-greedy test.

Reconstruct each root and verify RGB, goal and action history identity; use the
same frozen checkpoint and action bounds. Save every candidate, model score,
suffix and selection, along with call counts and source/model hash postchecks.
Same seeded CEM per prefix gives equal suffix search counts within this panel,
not an equal-total-budget comparison; inherited spectral search, encoding,
action encoding, internal model work and complete wall costs are not certified.
Run neural calls only on host CUDA under the existing cooperative lock.

If a model-selected prefix exists, it can be cross-referenced with earlier
physical outcomes only as an already-open exploratory comparison. Any
subsequent fresh physical intervention and independent confirmation need
their own fixed contract and cannot choose candidates by those outcomes.
