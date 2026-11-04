# Saved-prefix coverage: post-hoc result

One task met the outcome-dependent selection rule: the multistart CEM
selected chain ended at distance 0.034729, while v17 ended at 0.500235.
This is a retrospective diagnostic, not a fresh task or a performance
comparison.

Using the saved CEM chain as the reference, the nearest generated v17
prefix had action RMS distances 0.464, 0.386 and 0.414 at 5, 10 and 15
steps. The nearest prefix survived at depth 0 but was pruned at depths 1
and 2; the nearest retained distances were 0.464, 0.411 and 0.431. The
v17 final selected chain had action RMS distance 0.467 from the CEM chain.

This indicates that the saved v17 candidate tree did not reproduce an
action sequence close to this particular reference and that nearest-action
candidates were not always retained. It does not establish that the
pruned chains were useful, that another dissimilar chain could not solve
the task, or that pruning caused the failure. The saved branch and its
reference are not fed back into the method.

Protocol: `PC_V17_MULTISTART_SAVED_PREFIX_COVERAGE_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Machine-readable result:
`le-wm-main/results/pc_v17_e0_multistart_baseline_dev_20261104_codex/saved_prefix_coverage_posthoc.json`.
