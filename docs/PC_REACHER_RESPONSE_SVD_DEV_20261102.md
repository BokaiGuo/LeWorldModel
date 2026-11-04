# Reacher common-tail response SVD: development protocol

This is a new, outcome-informed development experiment on the two already
exposed roots 20263160 and 20263161. It is not an independent P1 test.
The fixed goal metric, checkpoint, action chart, warmup and five common tails
are inherited from the previously authenticated preflight.

At the neutral prefix, subtract tail-zero's predicted task-weighted terminal
coordinate from each of the other four terminal coordinates. Differentiate
this vector with respect to the ten prefix parameters using ten basis JVPs.
Project the resulting matrix off the instantaneous model-Q gradient, and
take its two largest nonzero right singular directions. For each direction,
evaluate both signs and fixed parameter radii 0.05, 0.10 and 0.20, mixed with
0.25 of the immediate-cost normal; evaluate the neutral prefix too.
Check the leading derivative with a symmetric finite difference. A degenerate
normal, response, failed derivative check or runtime error fails the root;
there is no fallback. The historical strict common-tail selector is unchanged.

All proposals and fixed-tail scores must be saved, with normalized and raw
actions, frozen source/model hashes, root and goal identities, errors and
post-run checks. Roots and search constants are fixed before execution. No
physical outcome is read by the selector. Model-only selection does not
establish an actual response change, non-greedy physical behavior, or P1.
The JVPs, encodes, SVD and model operations are not yet fully cost-metered;
this is not an equal-total-budget comparison. Execute only on host CUDA under
the cooperative GPU lock. A completed run may motivate, but cannot substitute
for, a separately frozen physical intervention and independent validation.

The first execution used the default CUDA attention backend and failed on
both roots before producing candidates because its efficient-attention
backward lacks the derivative needed by reverse-over-reverse JVP. That
failure is preserved. A separately sealed retry scopes PyTorch's math SDPA
backend to the compiler so the required derivative can be evaluated; it does
not change the proposal rules or selection conditions.
