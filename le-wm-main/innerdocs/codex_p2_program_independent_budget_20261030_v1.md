# Independent per-prefix resource ceilings

This new exploratory contract replaces the earlier shared-Jacobian action
intervention as the candidate for an independent-optimization comparison.
For each of four already known prefixes and each arm, independently recompute
the root Jacobian, SVD, candidate bank and winner. Both arms receive the same
per-resource ceilings: 36 packet / action_encoder / predict / predictor /
pred_proj, one Jacobian, one SVD and 17 terminal scores, plus 450 seconds.
Calls are upper limits, not quotas to fill. Actual attempted and returned
counts and individual wall time are retained, with no cross-arm credit.
Action-encoder/predict/predictor/pred_proj are nested and must not be summed.

The action-library arm composes separately named, fixed spectral primitives;
the baseline evaluates joint full10 spectral proposals. Their program spaces
and overhead differ; equal ceilings cannot establish equal FLOPs, energy,
wall time or total compute. Duplicate action sequences remain and their actual
unique count is reported. The source feedback and inherited acquisition/model
loading costs are common or separately recorded, not declared zero.

The source is an orchestration seam, not a production entry: the model scope
must wrap actual calls and supply synchronized time, and it must run inside a
durable supervised CUDA child with fixed sources and an independent saved
consumer. The fake adapter in tests makes no neural calls. No physical plan or
scientific admission is implied.
