# v12 contact-goal fixed-bank scoring diagnostic

## Findings

The fixed post-hoc bank contains 56 evenly indexed complete chains from the
sealed first v12 decision, plus eight separately labeled transformations of
the evaluator-only witness. All 64 were physically replayed, re-scored by
the frozen predictor on the same initial context, and scored from their actual
terminal RGB using the frozen encoder. No actions were returned to a planner.

- The 56 sampled planner chains produced **no successful physical endpoint**
  and all had the same object-position and object-angle error. Thus this
  subset provides no object-pose ranking comparisons.
- The unmodified privileged witness succeeded physically. Its re-evaluated
  model MSE was **0.1605**, versus **0.5583** for the best of the 56 sampled
  planner chains. The minimum original model score among all **484** saved
  full-horizon evaluations was **0.5537**. This large observed gap holds
  despite different evaluation batches; the largest original-versus-rescored
  difference on the 56 shared chains was `1.20e-7`.
- The witness is an evaluator-only trajectory generated using physical state;
  it was never part of the compiler's bank. Its favorable model score does not
  establish that the compiler could discover it or that a method succeeds.
  The observed result points toward a candidate-coverage/search issue on this
  root, while not excluding model error, other missing candidates or interface
  problems.

Among the eight witness-derived variants, predicted versus actual-encoded
MSE agreed on **27/28** non-tied pairs. Predicted MSE and physical object error
agreed on **17/18** non-tied pairs, with 10 ties. The full 64-chain bank has
only **four distinct exact object poses**, so these highly dependent pairs
are not independent evidence of general contact scoring. The actual terminal
encoding of the unmodified witness had MSE about `9.5e-11` against its
constructed goal image.

## Audit and limitations

An independent script reconstructed the planner slice and all eight witness
transformations, checked source/context and worker input hashes, independently
recomputed ordering arithmetic and repeated all 64 physical replays; RGB and
physical states matched exactly. The physics implementation and frozen neural
models are the same as the original experiment, and the neural outputs were
not independently recomputed by this audit.

This is one exposed constructed root, not evidence about the stopped random
goal, PC2 mechanism, PC3 planning advantage or independent generalization.
It cannot determine which individual stage of candidate generation, pruning
or optimization caused the absence of a good chain. The panel remains stopped.

Protocol:
`docs/PC_JEPA_V12_CONTACT_SCORE_BANK_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Raw bank, predictions, terminal frames, isolated encoding and audit:
`le-wm-main/results/pc_jepa_v12_contact_score_bank_posthoc_20261104_codex/`.
