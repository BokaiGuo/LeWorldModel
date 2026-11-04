# Wall prefix coverage, pruning and continuation: completed diagnosis

The fixed protocol ran on all eight exposed tasks. All 33 state-blind
prefixes received identical 128/256-candidate suffix optimizers, totaling
528 prefix/budget suffix plans. Three retention policies were sealed before
suffix optimization; their 48 selected full chains were replayed.
The all-bank census is diagnostic, not an executable arm's free search.

## Actual selected-chain results

| Policy | 128-candidate mean distance | Successes / 8 | 256-candidate mean distance | Successes / 8 |
| --- | ---: | ---: | ---: | ---: |
| Immediate greedy | .500537 | 0 | .500364 | 0 |
| Fixed-family lookahead | .500861 | 0 | .500265 | 0 |
| Random retention | .448790 | 0 | .417184 | 1 |

Success threshold .05 was fixed in advance. Lookahead is nonlinear
evaluation of a fixed suffix family, not spectral decomposition or a
demonstrated ability measure. Its screening cost is greater.

## Direct pruning evidence

The all-bank CEM census found a successful prefix at three tasks for each
budget, six task-budget settings total:

| Task | Budget | Successful bank index | Greedy retained | Lookahead retained | Random retained |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 256 | 16 | no | no | yes |
| 1 | 128 | 16 | no | no | no |
| 2 | 256 | 8 | no | no | no |
| 6 | 128 | 20 | no | no | no |
| 7 | 128 | 20 | no | no | no |
| 7 | 256 | 20 | no | no | no |

Those exact prefixes have identical saved CEM suffix outcomes regardless
of retention arm. Thus, for these settings, coverage and continuation were
sufficient in the full bank, but greedy/lookahead pruned the useful prefix.
This is stronger than merely observing a poor endpoint; it is not yet an
independent causal-mechanism or fair-performance result.

## Separately sealed post-outcome representability diagnostic

After observing the panel, a separate privileged diagnostic constructed a
straight-to-goal suffix from each exact prefix state using the same
two-turn/eight-translation template. It was not passed to any selector or
optimizer and did not alter prior results.

It found a successful representable prefix in **every task**, respectively
indices 16,16,8,8,28,28,20,20. Greedy and lookahead retained none of these;
random retained two (tasks 0 and 5). This is a witness of available solutions,
not a proof that no other useful prefix exists.

For these witnessed prefixes, CEM found a successful suffix in only 6/16
task-budget settings. Task 5 is particularly diagnostic: random retained
index 28, its template has a successful analytic witness, but CEM missed
success at both budgets. Therefore **retention and continuation search both
have demonstrated gaps**. It is not necessary to hypothesize that the bank
contains no feasible detour for this panel.

## Mechanism and accounting boundaries

Limited selected-suffix/own-planned-suffix witnesses were greedy 0/0,
lookahead 0/1, random 3/4 at caps 128/256. These compare actual selected
prefixes against zero and immediate same-bank greedy. A witness need not
mean task success. These are exact-model diagnostics, not real RGB evidence.
The full common fixed-suffix response tables are saved for every prefix.

Hypothetical per-arm full-chain rollout counts (screening and selected
continuation, excluding diagnostics):

| Cap | Greedy / random | Lookahead |
| --- | ---: | ---: |
| 128 | 1145 | 1420 |
| 256 | 2169 | 2444 |

Each rollout executes 16 transitions. Probe calls reused by lookahead are
not charged twice. The actual all-bank diagnostic also optimizes discarded
prefixes and separately replays interventions; it is not free method work.
No equal-total-resource or end-to-end hardware-cost claim is made.
Different CEM population sizes are not nested samples, so a previously
successful prefix can fail under the larger nominal budget.

## Next implementation target

Preserve these results. In a new development contract, test retention that
explicitly preserves diverse successor/action-response possibilities and
suffix search with turn-space coverage or independent multiple starts.
Change retention and continuation separately before combining them.
Keep ordinary greedy, random retention and no-SVD controls, and pay all
screening costs. Do not initialize from the privileged diagnostic witness.

This task implements a hand-designed action-program diagnostic, not
dynamics-generated programs. It does not validate SVD-specific value,
visual-model deployment, independent generalization, or P1 admission.

Artifacts:

- `docs/PC_WALL_RETENTION_SIDE_PROTOCOL_20261103.md`
- `le-wm-main/experiments/spectral_reachability/pc_wall_retention_side_20261103.py`
- `le-wm-main/results/pc_wall_retention_side_20261103.json`
- `docs/PC_WALL_RETENTION_COVERAGE_AUDIT_SIDE_PROTOCOL_20261103.md`
- `le-wm-main/experiments/spectral_reachability/pc_wall_retention_coverage_audit_side_20261103.py`
- `le-wm-main/results/pc_wall_retention_coverage_audit_side_20261103.json`
- `le-wm-main/tests/test_pc_wall_retention_side_20261103.py`
