# P2 B0 candidate reuse and prospective slot boundary

## Why the B0 actions repeated

CPU saved-only audit re-authenticated the shared-RGB model result and both
historical matched-model and physical-reuse receipts. The old panel case and
new capture-derived model input have exactly equal numerical latent history,
action history, goal and layout. Their context JSON hashes differ because
the old object additionally contains `context_length=3`, `latent_dim=192`,
and `action_dim=10`; this is not a difference in the numerical model input.

Both arms have exactly equal saved root Jacobians. For each arm all 16
candidate action byte strings and all 16 model costs match the earlier
multistart run, with winner index 12 in both runs. The comparison is made
within each arm across the two runs; it does not mean the two arms choose
identical actions to one another.

The fixed rule uses the canonical root-Jacobian SVD, four prefix anchors and
two local singular directions with both signs; it has no random restart
seed or newly sampled bank. The prefix-chain version repeats complete-prefix
evaluation to equalize instrumented packet calls; this bookkeeping change
does not introduce new candidate perturbations. Equal input and this fixed
construction explain the observed reuse; the exact candidate/cost comparison
is the stronger direct evidence. SVD sign fixing is not a guarantee of
cross-platform behavior inside a degenerate singular subspace.

This is not evidence that a common *strong optimizer* was run. The checked
comparison is `joint_multistart` against `prefix_chain` with a fixed
16-candidate spectral construction; it is not the separately specified
`no_composition_sra` arm and does not establish parity against a converged
CEM, gradient optimizer or GRASP. Changing the optimizer, candidates or
budget now would create a different exploratory contract and require
pre-outcome freezing, not a continuation of this exact comparison.

## Fixed roster and prospective source boundary

The old matched contract has four fixed cases and both arms already have
saved model and physical records. The verified integrity census finds only
two known root hashes; bases 2 and 3 duplicate the same source and both
actions. B0 was also re-executed by the immediately preceding run. Thus none
of these four is an unrun slot under the same contract.

The prospective queue document supplies a schema and synthetic test examples,
not a sealed actual slot roster. The checked results tree contains no named
P2 prospective queue artifact. The queue explicitly requires independent
P1 evidence, fresh-source/exclusion evidence, explicit MAIN authority, a
frozen plan, and an execution implementation; it does not authorize dispatch.
This limited census cannot establish that no eligible developer asset exists
anywhere in the workspace. It establishes no eligible unrun slot in the
inspected fixed roster or a verified matching prospective queue. No root was
chosen based on physical outcomes, and no P1 asset was repurposed.

For a future same-style *development-only* pair, the source must be fixed
before outcome inspection and bind actual root/reset, goal, observed RGB at
5/10/15, all 15 returned prefix actions and an exact native feedback guard;
the old source must not be represented as independent. Common setup is four
RGB encodes once; each arm is 16 candidates with packet/action encoder/
predict/predictor/prediction projection 44 each, Jacobian 5, SVD 5 and score
17, with per-arm 450-second cooperative cap. Physical replay has two fresh
environment instances, up to 70 public steps total including 20 warmup and
50 plan steps, 708 nested physics steps and 222 renders; preserve actual
attempts/returns and stop without replacement. Physical and model resources,
shared source acquisition and unknown total FLOPs/energy/I/O must remain
separate. This describes the existing engineering boundary, not an
authorization, equal-total-compute proof, P1 gate override, or a complete
prospective scientific contract.

The available execution path also needs a new source-specific capture and a
SHA-bound bridge joining that capture, the two new model winners and a genuine
native reset/feedback reference; the current B0 builder and physical bridge
are fixed to B0 and cannot simply be pointed at another slot. The independent
P2 queue remains blocked until its existing P1 and authority requirements are
satisfied.

## Verification

`p2_candidate_reuse_audit_20261102_exclusive` ran against the pinned real
saved assets and returned both arms with `candidate_action_and_cost_matches=16`.
Its two CPU tests passed, including altered latent, action and score rejection.
No model, GPU or physical execution occurred in this audit.
