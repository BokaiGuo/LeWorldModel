# Fixed-action history diagnostic at CP5

Use all 16 sealed CP5 feedback branches and the unchanged eight-action tail
bank. At CP5 compare (a) the source single-frame RGB context and (b) the two
real RGB contexts at steps 0 and 5 plus the actual executed five-action
prefix, with raw proprio fixed to zero in both. For (b), encode the executed
prefix and candidate tail with the same full-archive action normalizer, and
pass their two action embeddings aligned with the two visual context frames
to the frozen predictor. Do not use the wrapper's multi-frame zero padding.
This is a separately implemented exploratory interface hypothesis, not proof
of the original checkpoint's training data provenance.

Score the identical eight tails under each context. Recompute the original
single-frame costs and require agreement to within `rtol=1e-4, atol=1e-6`;
check initial and CP5 RGB hashes against the sealed source. For the action
actually executed in the source run only, compare each predicted pooled
step-10 latent against the sealed actual RGB-encoded step-10 latent. Record
all costs and both predictions. The history-selected counterfactual tail is
not physically executed in this diagnostic and its actual outcome is unknown.
Do not reselect source actions, treat branches as independent samples, modify
the checkpoint, or use simulator state to choose an action.

This is post-outcome development on four exposed roots, with unknown training
membership, development-calibrated camera and full-archive normalization.
It tests an input-interface hypothesis, not full causal attribution, a
properly budget-matched benchmark, or P1 confirmation.
