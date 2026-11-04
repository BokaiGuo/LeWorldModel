# Reacher saved-suffix feedback wiring

The existing two-stage physical carryover runner reconstructs the upstream
imagined suffix by rerunning CEM, executes the fixed prefix, and compares the
old tail and newly generated plans under the actual successor RGB and action
history. It now also accepts the suffix parameter witness saved in the
per-prefix selector receipt. For a saved witness, it verifies proposal index,
shape and finite values, recomputes its score from the imagined successor and
checks against the recorded cost before any physical action. This additional
verification incurs one suffix rollout, or two predictor samples for the
current two-packet suffix. Legacy receipts without a witness retain their
original CEM reconstruction path.

The saved tensor contains unconstrained parameters; the existing scoring and
physical executor apply the normal bounded action mapping. The existing
carryover selector includes the old tail in its fixed-size candidate batches,
so the tail changes the candidate population and elite updates. The separate
fixed-pool variant preserves the old population but adds a charged rollout;
these are distinct experimental contracts and neither establishes equal total
compute. Prefix compilation, verification, encoding, physical execution and
wall time also need to be included in any complete comparison.

Nineteen tests passed, including witness integrity, legacy fallback, successor
context binding and the existing carryover/fixed-pool contracts. These tests
use fixtures; the new saved-witness path has not yet been run on CUDA with a
newly generated sealed selector receipt, nor physically validated. No P1,
independence or planning-advantage claim follows from this wiring.
