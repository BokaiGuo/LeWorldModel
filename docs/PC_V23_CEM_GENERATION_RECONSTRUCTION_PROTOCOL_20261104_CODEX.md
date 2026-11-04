# v23 CEM generation reconstruction

Read the frozen eight-root paired E0 panel and its existing audit. For both
arms, independently reconstruct each optimization generation from recorded
initial actions, saved scores, the declared seed, and the implemented CEM
update rule. Compare every reconstructed bounded float32 sample to its
recorded action, check per-start and total allocations, and record pre-update
means and scales, elite-derived post-update means and scales, and incumbent
scores. Check that each start's updated distribution is used in its next
generation and that short final batches are accounted for.

This is an audit of recorded implementation and random-number replay, not a
new experiment or evidence that CEM is sufficiently optimized. The check
uses the same NumPy RNG implementation and recorded objective scores; it is
not an independent physical-environment replication.
