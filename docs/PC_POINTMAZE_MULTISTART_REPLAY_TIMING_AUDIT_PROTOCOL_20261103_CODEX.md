# Multistart physical replay and plan timing audit

For both completed multistart branches, replay all saved actions from their
original reset using the existing shadow bridge, verify all 18 images, and
obtain evaluator-only physical states from the same MuJoCo 2.1.2 worker plus
the authenticated 2.2 position backend. The worker requires four inputs, so
repeat the two trajectories once each and verify repeated results are equal;
duplicates are not observations. Compare final position to the earlier
spectral and single-start saved results on the same roots.

Separately inspect every saved five-action `plan_raw` from the original four
branches and the two multistart branches. Verify its first action matches
the actual dispatched action. For decision index `i`, compare the saved
remaining actions with only the following executed actions available before
the fixed 17-step endpoint. Report the number available, the number different
at tolerance 1e-6, and max raw-action L2 discrepancy. Earlier decision 0's
five-step planned terminal score can be compared descriptively with final
realized RGB-derived score, but because replanning changes actions it is not a
same-action prediction-error measurement. Decisions 1–4 predict beyond this
run's endpoint and must not be falsely labeled realized five-step outcomes.

All roots are development/exposed, outcomes are replay labels, and a two-start
control with 100 scored samples versus 50 plus spectral differentiation is
not an equal-total-cost comparison. Never change historical actions, budgets,
thresholds or success labels.
