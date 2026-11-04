# Feasible spectral-prefix backtracking: development result

The protocol `PC_REACHER_FEASIBLE_PREFIX_BACKTRACK_PROTOCOL_20261102.md`
was executed on the same two outcome-exposed roots using host CUDA. Frozen
model hash and root/goal RGB identity checks passed.

| Root | Retained source/alpha pairs | Selected bank index | Previous selected index |
|---|---|---:|---:|
| 20263160 | (2, 0.25), (5, 0.5) | 2 | 2 |
| 20263161 | (2, 0.125), (5, 0.25) | 5 | 5 |

The other two refined initializations on each root had no feasible point in
the prescribed interpolation grid. Unlike the preceding soft-penalty run,
this procedure generated two new model-feasible prefixes per root. The
unchanged predicted-successor CEM nevertheless preferred the original
selected action in both cases. This establishes only that the new candidates
did not win under this model score and this development budget; neither the
score nor the model's feasibility condition proves physical task behavior.

Each root paid the previous refinement cost of 17 prefix samples, 85 suffix
samples and 16 backward calls, plus 20 prefix and 100 suffix backtracking
samples. The final selector used 528 and 660 samples respectively, versus
396 and 528 in the original full-bank run. Encoding and total resource
equivalence remain uncertified, so this is not a fair performance comparison.

Because the selected actions are unchanged, no new physical experiment was
run and no independent evidence was obtained. Preserve the previous real
execution findings. Further work needs either better candidate generation or
a task-aligned selector, followed by fresh, predeclared physical comparisons;
changing the gate or cherry-picking these candidates would not establish P1.

Artifact: `le-wm-main/results/reacher_pc_feasible_prefix_development_20261102_v1/report.json`.
The refinement, backtracking, guarded compiler and selector regression suite
passed 18 tests; synthetic tests do not establish physical efficacy.
