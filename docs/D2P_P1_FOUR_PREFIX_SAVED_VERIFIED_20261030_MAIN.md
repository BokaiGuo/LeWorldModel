# Four-trajectory prefix intervention: saved verification

The previously executed exploratory fixed 24-branch intervention has now
passed its separate saved-data audit. It records 22 complete 15-step
endpoints, one out-of-domain censored branch and one native early stop.
Neither missing endpoint is imputed; all 24 branches and the 28 prescribed
comparisons remain represented.

For the structured trajectory with the same saved suffix, the recorded
terminal object-position errors are 20.40 for selected, 44.50 for zero and
12.76 for greedy. This is a single known-root, post-outcome example, not
evidence that selected beats greedy or that a population-level control
capability changed.

For the same structured trajectory, comparing each prefix's saved suffix with
its own zero suffix gives position-error changes of -50.38 (selected),
-26.28 (zero), and -4.59 (greedy). The selected and zero prefixes have the
same recorded object-position error of 70.79 under the zero suffix, whereas
greedy is already at 17.35. Thus the fixed saved suffix produces a larger
relative object-position improvement after selected than after these two
comparators, while greedy still has the best absolute endpoint. This is an
exploratory within-instance contrast: the saved suffix came from the selected
trajectory, and subtracting the zero-suffix endpoint does not establish a
general capability advantage, remove all passive-dynamics effects, or compare
each prefix's independently optimized suffix.

The independent audit covers the saved journal chain, operation receipts,
fixed action execution, returned states, image hashes, guard and branch
statuses. It does not reauthenticate the H10 upstream source closure or prove
runtime authenticity. There is no independent confirmation, equal-compute
claim or P1 admission.

- Result: `le-wm-main/results/codex_four_prefix_saved_audit_20261030_CODEX02.json`
- Result SHA-256:
  `ea1a3a7480c5058c960fbb2e81338a87dfad8e2d8e5f1642719e8ff9242ececa`
- Physical report SHA-256:
  `d8086e1da759544af775974f0b161d5de432cca5fbc445109727f1e73ab2bdf7`
- Consumer tests: `tests/test_codex_four_prefix_saved_audit_20261030.py`,
  7 passed.
