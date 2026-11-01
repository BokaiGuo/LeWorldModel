# Conditional H2: successful CUDA run and independent saved-data audits

Actual source run:
`results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json`.
SHA256: `64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1`.
All eight arms completed on host RTX 5060 Ti CUDA: 528 predict entries,
528 action encoder entries, 88 fresh Jacobians, 96 SVD calls and 16896
mathematical output VJP directions. Backward engine API count is unknown.
Model unchanged, failure null, failures empty. These are four used calibration
roots, not an independent confirmation cohort or complete research closure.

## Main independent arithmetic

`p1_conditional_mechanism_arrays_main_20261027_v1.py` validates the source export,
rebuilds all banks and suffix solves, and describes selected-vs-greedy costs
within each metric. All eight selected prefixes equal their own stage-greedy
selection. Thus this run has no selected strict non-greedy prefix witness. All
four old/new-Q program pairs have different raw actions; scalar costs from
different Q are deliberately not subtracted as a shared utility. This audit
performs 192 saved-array SVDs, zero neural or physical operations.

`p1_conditional_history_arrays_main_20261027_v1.py` independently reconstructs
normalized action tokens in NumPy float32 from the sealed AST-literal public
mean/std, avoiding imports of model and physics runtimes. It strictly checks
layout metadata, both action tokens and finite three-state latent shape for
80 candidate contexts, 24 repeated selected/replay contexts, and eight exports.
The actual result has all 112 content checks true, bit-exact to the saved
float32 token values. This audit performs 96 saved-array SVDs in upstream
export validation, zero neural or physical operations. It does not certify the
public constants' training-data provenance.

Ten joint tests of these two main auditors pass. Algebra/mock tests do not
establish model prediction quality, actual capability change, task success or
P1 admission. Both actual audit outputs are in same-stem `results/*_main_v1`.

## Remaining implementation findings

The independent compiler review found failure-finalization and partial-array
retention gaps, plus a missing action-history-content check in the original v1
verifier. The successful run did not encounter the first two fault paths. The
history audit above verifies this actual run, but does not patch the frozen v1
verifier. An independent v2 repair is required for future compiler executions;
all old sealed source and output bytes remain unchanged.

Fresh physical execution of these exact programs, independent P1 roots,
non-greedy mechanism, full matched strong baselines, and downstream scientific
admission remain open. No threshold, beta, task success definition or actions
were changed based on these observations.
