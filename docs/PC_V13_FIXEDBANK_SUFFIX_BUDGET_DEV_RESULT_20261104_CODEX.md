# v13 fixed-bank suffix budget development result

The new four-root exact-Wall diagnostic completed all 40 candidate pairs.
Each fixed first-layer v13 candidate received a 120-proposal search and a
three-start search of 480 proposals each. The first 120 actions in the
first large-search stream matched the small-search actions. Both use the
same zero-tail incumbent; generation was done once per root.

| Root | Best of all 10, 120 | Best of all 10, 3 × 480 | Best retained, 120 | Best retained, 3 × 480 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.500083 | 0.500083 | 0.502675 | 0.500179 |
| 1 | 0.500639 | 0.500085 | 0.500639 | 0.500085 |
| 2 | 0.500448 | 0.500059 | 0.500448 | 0.500059 |
| 3 | 0.500803 | 0.500055 | 0.500803 | 0.500055 |

The larger search strictly improved 38/40 individual candidate scores, but
neither budget found a chain below the preexisting 0.05 success threshold:
0/4 roots for the full bank and for the retained subset. This does not show
that continuations do not exist. It limits the hypothesis that increasing
this particular four-knot CEM budget and adding two starts is sufficient on
these four tasks; full-action parameterization, other optimizers, candidate
coverage, and dynamics remain possible explanations.

The separate audit checked all 62,440 saved full-chain suffix trajectories
against the alternate same-environment geometry, fixed-prefix identity,
proposal counts, best score and nested stream. The suffix diagnostic used
187,320 five-step prediction tokens, excluding compiler generation;
per-root compiler generation meter records are in `result.json`. These
costs are not a full-method, equal-resource comparison.

## Trace limitation

The protocol asked for every action to be saved. The runner saved every
suffix-diagnostic action, but **did not save the compiler-generation
prediction batches**: it saved the generated layer and the generation
meter, not the complete generation trace. Thus the alternate-geometry audit
covers the suffix diagnostics, not every v13 generator call; this result
does not meet the protocol's full trace requirement. The existing v13
regression tests passed 11/11, but they do not repair that missing trace or
establish a scientific gate. Any future protocol requiring complete
generation replay needs a new runner and new protocol; do not rerun or
replace these four roots.

This is four exposed development tasks from one exact environment, with
broader provenance unknown; it is not independent replication, a PC2
same-chain intervention, a PC3 comparison, or RGB control evidence.

Protocol: `PC_V13_FIXEDBANK_SUFFIX_BUDGET_DEV_PROTOCOL_20261104_CODEX.md`.
Raw manifest, per-root traces, records and audit:
`le-wm-main/results/pc_v13_fixedbank_suffix_budget_dev_20261104_codex/`.
