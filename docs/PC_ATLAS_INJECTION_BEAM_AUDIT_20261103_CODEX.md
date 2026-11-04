# Atlas strong-chain injection: actual beam coverage audit

## Question and method

This audit supplements the existing oracle-injection result; it does not replace
or modify it. The saved eight gradient chains and goals are replayed through an
instrumented subclass of the existing v2 search. The subclass records candidate
banks without changing their actions, order, random generator, or search logic.
For each goal, the audit independently replays every candidate in the root bank
and in the injected successor's suffix bank. It checks the full-context direct
and split hashes against the existing saved witness, as well as the saved branch
terminal cost and selected root.

The v2 search expands **every root prefix separately**. At depth 2, each
branch's seven terminal candidates are sorted by model cost and truncated to
beam width 2, after which only its first candidate is returned. The other
survivor and discarded candidates are not exposed in the ordinary result. The
audit reconstructs their ranks from the recorded bank and deterministic backend.
Ranks use exact floating-point costs and stable bank-order tie breaking, matching
the Python stable sort. This reconstruction is not a direct trace of the internal
list after sorting.

## Findings

| Goal | Injected prefix rank among 7 | Injected suffix rank among 7 |
| ---: | ---: | ---: |
| 0 | 5 | 1 |
| 1 | 3 | 1 |
| 2 | 3 | 1 |
| 3 | 4 | 1 |
| 4 | 4 | 1 |
| 5 | 4 | 1 |
| 6 | 3 | 1 |
| 7 | 3 | 1 |

All eight suffixes rank first and therefore survive the actual width-2 terminal
beam, win their branch, and are selected overall. Thus the prior test **does
execute the beam sorting/truncation path**, but never tests a case in which the
injected chain is near the cutoff or actually discarded. At depth 2 this is a
terminal ranking, not an intermediate-node pruning test; no depth-3-or-greater
claim follows. Ordinary non-injected alternatives are discarded, but this
audit does not instrument or independently verify their internal discard events.

As a counterfactual only, ranking these *observed* root banks by prefix cost
would drop all eight injected roots at global width 2; width 3 would retain
four, and width 4 seven. Actual v2 does not perform this global pruning. These
numbers do not constitute a rerun of a global beam: changing the search can
change subsequent random-number consumption and candidate banks.

The old 90 rollout and 8 Jacobian compiler calls per root are reproduced. The
audit adds 17 direct backend rollouts per root: one constructor successor,
seven root-bank scores, seven suffix-bank scores, and two full/split checks.
They are outside compiler counters. Gradient optimization is not rerun; the
previously saved privileged actions are inputs. This is deterministic
exact-dynamics, exposed-development model scoring, not independent physical
evidence, an equal-budget experiment, or Atlas performance.

## Files and verification

- Existing source: `le-wm-main/results/pc_atlas_oracle_injection_20261103.json`.
- New audit: `le-wm-main/experiments/spectral_reachability/pc_atlas_injection_pruning_audit_20261103_codex.py`.
- New record: `le-wm-main/results/pc_atlas_injection_pruning_audit_20261103_codex.json`.
- New tests: `le-wm-main/tests/test_pc_atlas_injection_pruning_audit_20261103_codex.py`.
- Targeted new and existing injection/search tests: 10 passed.

Reproduce from the repository root:

```sh
PYTHONPATH=le-wm-main python -m experiments.spectral_reachability.pc_atlas_injection_pruning_audit_20261103_codex --input le-wm-main/results/pc_atlas_oracle_injection_20261103.json --output NEW_OUTPUT.json
PYTHONPATH=le-wm-main python -m pytest -q -p no:cacheprovider le-wm-main/tests/test_pc_atlas_injection_pruning_audit_20261103_codex.py le-wm-main/tests/test_pc_atlas_oracle_injection_20261103.py le-wm-main/tests/test_d2p_atlas_search_v2.py
```

The output path must not already exist. A future intermediate-pruning test
requires a separately designed depth-3 witness and explicit instrumentation;
it cannot be inferred from these terminal winners.
