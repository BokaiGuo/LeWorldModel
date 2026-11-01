# Real spectral bank as a common-suffix initializer: wiring plan

Status: design only; no new spectral evaluation, CUDA, physics, or claims of
fairness. The existing four-arm host entry remains
`local_ridge_proxy_not_D2P_spectral_bank`; this plan does not rename it.

## Existing implementation and semantics

The existing `agent_p3_r_spectral8_bank_20261029_v1.py:37-47,156-209`
provides an actual spectral8 bank: SVD of float64 `Q @ J_raw`, canonicalizes
the sign of each right singular vector by its first largest-magnitude entry,
then uses two modes, two fixed radii and signs in radius/mode/sign order.
It maps logits through float64 tanh and casts once to float32. H3 is an
explicit 30-dimensional extension under the same total radius; the existing
contract does not claim equivalence to the original R or authentication of
the resulting bank. A separate zero control is not one of the eight.

`pusht_full_packet_compiler_20261027.py:60-80` is a *different* bank:
zero + DLS progress + eight spectral candidates, with an additional SVD
inside its bounded ridge solve. Do not interchange the two methods or charge
only one SVD for that ten-candidate compiler. For the proposed wiring choose
the former explicitly, retain its `spectral8 H3 extension` label, and call it
an exploratory candidate initializer, not an established D2P baseline.

The spectral8 builder currently accepts a parent-bound saved signal and an
original `core.Ledger`, and performs nested program-compile/array-SVD calls.
It does not generate a Jacobian, independently authenticate its artifacts,
or run the neural model. The current four-arm durable driver forbids nested
dispatch. Therefore simply wrapping `bank.build()` inside a four-arm dispatch
would fail or double count; importing an old Jacobian for free would omit its
acquisition cost. No existing bank has been asserted to be a fresh, paid
signal for this fixed four-arm execution.

## Proposed prospective implementation

1. Fix and hash one new root/task/checkpoint/precision/source closure, the
   spectral8 H3 rule and radius constants, all four initializers, seed,
   candidate roster and ordering, a common suffix configuration, score/ties,
   and per-arm resource vector *before* any outcome. Fix eight candidates for
   each arm (or separately preregister a rigorously justified matching rule);
   do not pick spectral modes, radii, root, or candidates after scores.
2. In the structured arm, acquire the 192-by-30 raw-action Jacobian at the
   fixed nominal raw-zero H3 path using the same frozen model and normalized
   history. Meter the three action-encoder/predict calls and each of 192
   VJP rows on the durable driver; fail on any nonfinite value or exception.
   J is already differentiated through normalization; do not divide it by std
   again. Bind the measured nominal, J, Q and producer inputs/bytes.
3. Perform `Q @ J` and the actual float64 SVD once under a non-nested metered
   SVD dispatch; record inputs, full output, ordering, signs, degeneracy and
   software/device details. The existing four-arm vector has `svd_calls` but
   no matrix-multiply, decomposition size, CPU time or data-transfer axes:
   add explicit prospective accounting or a complementary measured log before
   treating this as all-cost coverage. A degenerate singular subspace is not
   uniquely fixed by sign canonicalization; a frozen tie/degeneracy policy and
   numeric acceptance criterion must be specified before production.
4. Persist all eight generated raw `[3,5,2]` candidates and trace the exact
   bytes before they are scored or optimized. Feed each through the same
   `codex_four_arm_raw_suffix_20261030_main.optimize` algorithm and fixed
   settings; re-evaluate *all post-update* candidates under the same model
   replay and deterministic selection. Preserve input/output bytes and pay
   every candidate, including losers and failed attempts.
5. Each arm has its own durable ledger. The current leaf binds one optimizer
   config digest to the whole contract, while the existing host uses its CEM
   config and the raw-gradient optimizer expects exactly `steps`,
   `learning_rate`, `history_size`. A new composite immutable contract and
   adapter are needed to authenticate both initialization and suffix config;
   do not mutate old files or bypass the digest check. Fixed replay also
   currently binds three candidate IDs; a new eight-candidate contract and
   consumer/test are required.

For a simple sequential per-candidate version with eight candidates, H=3,
and S common gradient steps, structured model calls include 3 nominal
action-encoder/predict calls, then `8 * S * 3` for common optimization and
`8 * 3` for final replay; VJP/backward calls include `192 + 8*S`. There is
one actual spectral SVD plus one Jacobian marker, eight replay markers and
eight selection evaluations. Initializer costs of the other arms and any
extra screening must be added independently. These are prospective counts,
not measured costs or evidence that equal caps create equal total cost.
Budget must reserve final replay before expensive operations and fail closed
before a call that would violate a cap; never silently reduce the roster.

Required tests before offering a new launcher: exact spectral8 rule vs pinned
existing implementation on nondegenerate fixtures; sign/order/tanh and H3
shape; degeneracy and nonfinite rejection; analytic J with normalization
chain; post-update-not-input replay; eight-candidate trace and tie; per-arm
successful/failed/blocked call accounting, no nested dispatch and no unpaid
SVD, truncation/tampering and partial status. A separately authorized real
checkpoint host CUDA ABI test would still be needed; CPU fixtures are not it.
