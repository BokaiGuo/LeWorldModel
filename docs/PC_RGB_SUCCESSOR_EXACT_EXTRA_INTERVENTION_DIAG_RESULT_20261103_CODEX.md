# Extra selected-chain intervention: random bridge and lookahead

The post-outcome protocol
`PC_RGB_SUCCESSOR_EXACT_EXTRA_INTERVENTION_DIAG_PROTOCOL_20261103_CODEX.md`
compares the final selected successor first segment with the final selected
first segments from the sealed random-bridge and lookahead full-method
controls. Thus these include each control's joint refinement and fallback,
not merely its original unrefined candidate.

On all 24 exposed roots, the same six suffixes were replayed and each prefix
received the same 120-sample CEM replanning policy and root-specific seed.
The selected successor's actual suffix gave a more favorable change relative
to each prefix's own zero suffix than both extra controls on **21/24** roots.
With each prefix independently replanned, the successor had lower terminal
distance than both extra controls on **3/24** roots. These two counts do not
require non-immediate progress and cannot be combined across different roots.

The earlier, separately specified same-chain screen against zero and
same-bank greedy had an intersection of **0/24**, so these extra comparisons
cannot make that screen pass. Eight successor selected prefixes were zero,
following the previously executed model-score fallback. The new diagnostic
is post-outcome and does not establish causal reconfiguration, PC2, or PC3.

All common-suffix scores, zero-relative changes, 120 replan scores and action
chains are sealed in
`le-wm-main/results/pc_rgb_successor_exact_same_policy_dev_20261103_codex_extra_intervention_diagnostic.json`.
Independent tests reconstruct every prefix, suffix, score and seeded CEM
result; the related selection passed five tests.
