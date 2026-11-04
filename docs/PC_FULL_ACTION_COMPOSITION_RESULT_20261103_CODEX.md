# Full-action composition: exact-dynamics development result

The protocol was fixed in
`PC_FULL_ACTION_COMPOSITION_PROTOCOL_20261103_CODEX.md` before this run.
The 24 tasks are deterministic synthetic geometry tasks, not independent
training-disjoint or visual-model confirmation. No historical gate changed.

## Full-action programs and terminal distance

Each selected staged prefix has nonzero translation and turn on all 24 tasks
at each budget; the new prefix parameterization no longer forces zero
translation. This does not mean eight proposals exhaust the continuous action
domain.

| Objective cap | Spectral | Response without SVD | Random prefix | Flat full-action |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.170596 | 0.170596 | 0.161958 | 0.051292 |
| 48 | 0.123995 | 0.123995 | 0.161958 | 0.003872 |
| 96 | 0.123995 | 0.123995 | 0.161958 | 0.003759 |

Entries are mean final physical-position distances in the exact dynamics.
The staged arms each optimize the suffix for eight fixed proposed prefixes;
flat optimizes all three effective action coordinates. All use bounded
L-BFGS-B and the same two-step horizon. The response generators implement
the same regularized inverse using either SVD or a direct linear solve; the
largest paired spectral/no-SVD terminal-distance difference is `2.22e-16`.
Consequently this particular test shows no added value from the factorization
itself, not that SVD cannot be useful for other systems or algorithms.

The prespecified development criterion required spectral to beat both
no-SVD and flat by more than `1e-9` at two of three budgets. It met this at
zero budgets. **Pause this spectral construction** under the stated rule;
do not redefine it as an RGB mechanism result or a PC2/PC3 success.

## Cost and diagnostic limits

The spectral and no-SVD arms pay five two-step finite-difference/nominal
model evaluations from the cap, plus one SVD for the spectral arm. The other
two arms pay no such generation evaluations. Mean actual total evaluations
for spectral / no-SVD / random / flat were:

| Cap | Spectral | No SVD | Random | Flat |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 22.83 | 22.83 | 22.21 | 24.00 |
| 48 | 29.79 | 30.00 | 24.29 | 47.83 |
| 96 | 29.79 | 30.00 | 24.29 | 71.12 |

Early-converged per-start allocations were not redistributed, as specified
before the run. Thus the staged arms leave substantial higher-budget
allowance unused; the table shows actual use rather than implying matched
compute. The flat method uses its allowance to update both prefix and suffix,
while staged methods cannot refine a frozen prefix. The present comparison
tests these concrete methods, not an upper bound on all compositional search.
The operation types, SVD overhead and wall times are recorded separately and
are not equated to FLOPs. Tiny floating-point deviations can change the exact
number of optimizer iterations even though the SVD/no-SVD outputs agree to
roundoff.

For the spectral method, common-suffix active response relative to each
prefix's own zero suffix is more favorable than zero-prefix response on
22/24, 23/24 and 23/24 tasks at caps 24, 48 and 96. These counts are
descriptive and do not require a non-immediate prefix; they are not a
mechanism gate. In an exploratory stricter `1e-9` post-run tally, 10/24
spectral selections at each budget also meet non-immediate position and
separately optimized zero-prefix comparisons; only one at cap 24 and none at
the other caps beats the corresponding flat method. Those diagnostic counts
are not predeclared statistical tests, and this exact-state experiment is
not a real RGB intervention.

## Artifacts

- `le-wm-main/results/pc_full_action_composition_20261103_codex.json` contains
  all 24 tasks, candidate prefixes, complete plans, cost and response records.
- `le-wm-main/experiments/spectral_reachability/pc_full_action_composition_20261103_codex.py`
  is the new runner.
- `le-wm-main/tests/test_pc_full_action_composition_20261103_codex.py`: three
  tests passed, including full-prefix coverage, SVD/no-SVD equivalence,
  replay/score consistency, budget limits and the common-suffix comparison.

Training provenance, root independence and complete visual-run cost
certification remain unresolved on the separate PointMaze performance line.
