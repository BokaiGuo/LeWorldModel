# Matched-call nullspace candidate ablation

This new development ablation uses the eight already exposed unicycle goals
and seeds 42–44 from `PC_ATLAS_MATCHED_EXACT_DEV_20261103_EXCLUSIVE.md`.
Compare the saved random, nonzero spectral and ordinary-turn candidate arms
with a new task-response-nullspace arm. Every arm has the same shared zero
and progress prefixes plus two extra candidates, the same relinearized
one-packet suffix optimizer and full-chain witness replay.

The new arm reuses the root Jacobian already computed by the shared progress
operation; compute the full right nullspace of the same position-weighted
Jacobian, and add the positive and negative first null direction with raw
maximum absolute component equal to the existing prefix radius 0.5. If the
required null direction is absent, fail rather than substitute another
candidate. Both prefixes must satisfy action bounds. The random, spectral
and ordinary arms retain their original predeclared proposals, including
ordinary turns at their original amplitude. Thus action amplitudes and
candidate-generation computation are not matched.

Require 28 backend rollouts, five Jacobians, 32 rollout action steps and
five differentiated steps per arm. This is equality of measured backend
calls, not wall time or total compute: the nullspace arm requires an extra
SVD, while computing shared progress geometry for all arms. Bind the previous
saved result and script, compare each paired goal/seed, preserve all
nullspace-arm candidate actions, scores and witness hashes, and report
ties, outcomes and actual counts. The three seeds do not give three
independent goals.

The selected model program and model-based nonprogress are not evidence of
physical mechanism, independent generalization, neural control or a gate.
