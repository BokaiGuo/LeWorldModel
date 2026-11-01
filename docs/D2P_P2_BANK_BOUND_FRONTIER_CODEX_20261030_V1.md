# P2 bank-bound frontier: exploration, no physical authorization

`codex_p2_bank_bind_before_compare_20261030_v1.execute_native` executes the
existing CUDA frontier with the common-cap hooks. After the shared root
compiler and deterministic candidate bank run, it writes and fsyncs the
bank and its per-arm provenance binding before starting either comparison.
It verifies the bank and binding bytes before and after each arm and rejects
in-memory bank mutation. It uses the existing native all-candidate validator,
forward-entry and operation validators. No test initializes CUDA or runs a
neural model. The runtime-generated bank is not a claim of pre-registration;
the rule and input authority must have been independently fixed beforehand.

`codex_p2_bank_bound_saved_consumer_20261030_v1.consume` binds an externally
retained report digest to the bank, provenance and event bytes, checks
bank-before-arm order and original bank recipe, then uses the existing
forward, operation and all-candidate checks to derive raw10 plans. It is a
saved-data verifier, not remote attestation or proof of physical execution.
Its caller must separately verify the complete prepared protocol, actual
source closure, checkpoint identity and trusted external report digest. The
native runner does not itself load or authenticate a model, check its frozen
state, reserve a GPU or prove complete resources; those remain prerequisites.

## Physical execution remains blocked

The existing physical consumer's preparation is coupled to a fixed historical
planner report hash and four-root authority. Its root collection, 15-step
warmup, RGB/goal checks, per-step guard and saved consumer must be rebound to
the new report and bank artifacts under a fresh, independently approved
four-root-by-two-arm physical protocol. Current new saved plans are not that
physical protocol. No new real physical dispatcher is supplied or authorized:
reusing the old one unchanged would reject the new report, while bypassing its
identity checks would weaken the gate. Implement a new sealed preparation,
atomic cross-entry physical claim, fresh-environment execution and strict
saved-action consumer before requesting physical execution.

P1/P2 admission remains false. The 80 versus 40 candidate frontier has common
caps on selected resources but unequal Jacobians and VJPs; total FLOPs, I/O,
model loading, encoding and other resource costs are not fully covered.
Known roots are calibration, not an independent prospective cohort.
