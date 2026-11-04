# Fixed lattice CUDA development launch

On the same two exposed roots and frozen inputs as the preceding constrained
experiment, evaluate all 41 fixed candidates using the original common-tail
Q objective and strict selector. Seal the source, fixed protocol, preflight,
checkpoint, normalizer and goal-Q before execution. No physical outcome
enters selection. Preserve null selections and per-root errors.

Reevaluate all 41 candidates and record actions, endpoints, Q scores and
non-gating response census. Per root the search uses 41 prefix and 410
suffix-token samples; the census repeats those counts. Encoding, Q
construction and full internal model costs are not certified. The initial
candidate protocol incorrectly called 205 the suffix-token count: there are
five tails and two predicted suffix tokens each. This correction is made
before this model run, with no selection or gate change.

This is a development search, not an equal-budget comparison or P1 outcome.
