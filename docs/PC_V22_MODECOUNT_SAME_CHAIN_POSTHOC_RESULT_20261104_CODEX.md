# v22 two-mode post-hoc same-chain mechanism diagnostic

All eight previously selected two-mode chains were evaluated; neither the
zero/greedy mechanism intersection nor the full-control intersection held
for any of the eight. The separate predicate/action audit verified the
conditions and replayed the stored suffix and common-tail actions. This was
designed after the paired panel's outcomes were visible, so it cannot be
presented as prospective PC2 confirmation.

On the one successful planner root, the selected prefix was non-improving
at the intermediate step and the selected actual tail achieved terminal
distance 0.020313. Applying that same tail after zero and same-bank greedy
prefixes yielded 1.020068 and 1.981820. However, independently replanning
with the prescribed common CEM gave mean distances 0.503406 from selected,
0.500989 from zero, and 0.500595 from greedy. Consequently the prescribed
same-budget replanning condition was not met. These observations localize
the remaining issue to a combination of tail search and mechanism evidence;
they do not prove that the prefix has no useful successor, nor that any
implementation bug caused the outcome.

For an additional post-hoc expression check, the selected ten-step tail on
each root was least-squares projected into the intervention optimizer's
four-knot family and clipped exactly as legal actions. On the successful
root, the projected tail attained 0.006881: this demonstrates a successful
tail in that family for this prefix, while the prescribed optimizer did not
find one. It diagnoses search insufficiency at that budget on this root;
it does not make the privileged projection a planner candidate or establish
that the knot representation is adequate generally. This projection was
not optimized for terminal score and is not a bound on the family.

All results, including the seven failed planner roots, are retained under
`le-wm-main/results/pc_v22_modecount_same_chain_posthoc_20261104_codex/`.
The reconstruction uses the same intervention code and geometry; it is not
an independent replication.
