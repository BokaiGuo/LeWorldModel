# Depth-3 injected-chain intermediate pruning: development result

## Existing coverage and new experiment

The prior strong-action injection and follow-up audit covered depth 2: the
injected suffix was ranked at the terminal beam, not at an intermediate
pruning boundary. No same-purpose depth-3 injected-witness audit was found in
the Atlas documentation, code, and tests inspected for this task.

The separate protocol constructs two three-action chains from each of the
eight previously exposed gradient witnesses. `early` executes the original
two actions then zero; `delayed` inserts a small backwards action and adds the
same velocity back at the last action. Their terminal latent states agree to
within 1.12e-16 in the observed calculations; maximum difference in terminal
weighted model cost is 6.01e-22. This is a deliberately constructed
counterexample, not independent evidence or a performance comparison.

The experiment runs the **unchanged v2 search** at depth 3 and beam width 2.
A subclass injects actions at matching full-context hashes. Python line tracing
captures the actual expanded nodes and actual beam immediately after the
existing assignment; it additionally verifies identity and ordering against
the stable cost-sorted slice. Each record stores candidate names, actions,
scores, context hashes, ranks and retention at both depths.

## Observed intermediate cutoff

| Goal | Early chain rank | Early retained | Delayed chain rank | Delayed retained |
| ---: | ---: | :---: | ---: | :---: |
| 0 | 1 | yes | 5 | no |
| 1 | 1 | yes | 6 | no |
| 2 | 1 | yes | 5 | no |
| 3 | 1 | yes | 4 | no |
| 4 | 1 | yes | 4 | no |
| 5 | 1 | yes | 4 | no |
| 6 | 1 | yes | 5 | no |
| 7 | 1 | yes | 5 | no |

All eight delayed middle nodes are actually dropped before the third packet;
their named terminal continuations are never offered at the matching context.
This is not a simulated global-root cutoff: v2 expands root prefixes
separately, and the trace covers the injected root branch.

For goal 0, the delayed intermediate model cost is 0.102119, rank 5 among
seven; the two kept costs are approximately 0.00000407 and 0.00000828. Its
unexecuted, directly replayed three-action witness has terminal cost
1.24e-17, while the best actually returned branch in that injected run costs
3.99e-10. This illustrates that intermediate model cost is not a lower bound
on a supplied continuation's terminal cost. It does **not** prove a bug:
finite-width beam search is designed to prune candidates, and the discarded
third action is privileged information.

## Path identity and tie handling

All eight early middle nodes survive, and their named third actions are
offered. However, the specifically named third action is rank 3 and is
discarded at the terminal width-2 cut in every case. It is exactly the same
zero action and reaches the same full context as an earlier ordinary `zero`
candidate, which ranks 2 and survives; `progress` ranks 1. Consequently the
**named** three-edge injection is not retained, but an action-identical
continuation is. The final branch winner uses `progress`, not the named
injected terminal action. This distinction is essential when auditing labels,
full-context identity and action witnesses.

Injection is keyed on full-context digest, not on path name. Some contexts are
visited by more than one branch; the JSON separately records context visits and
the exact named path. In these delayed cases the third-action context was
visited zero times after pruning. No claim is made that this must hold for
other chains or backends.

## Accounting, verification and limits

Each early run used 231 compiler rollouts and 22 Jacobians; each delayed run
used 230 and 22. Total: 3,688 attempted/returned compiler rollouts and 352
attempted/returned Jacobians. Another 80 backend rollouts construct and check
the 16 witnesses outside the compiler counters. Trace scoring calls, wall
time, gradient planner work that produced the saved inputs, and test work are
not included. These are not matched-compute runs.

The exact unicycle backend is the model, not a learned visual model or physical
environment. The eight exposed goals are not independent confirmation. These
privileged chains do not measure Atlas performance, general pruning error
rates, physical control, or any admission gate. Python tracing requires
exclusive tracing and the expected v2 assignment; it fails closed otherwise.

New files:

- Protocol: `docs/PC_ATLAS_DEPTH3_INJECTION_PRUNING_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_atlas_depth3_pruning_trace_20261103_codex.py`
- Trace and results: `le-wm-main/results/pc_atlas_depth3_pruning_trace_20261103_codex.json`
- Tests: `le-wm-main/tests/test_pc_atlas_depth3_pruning_trace_20261103_codex.py`

The new and existing relevant tests report **16 passed**. Reproduce using a
fresh output filename:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main python -m experiments.spectral_reachability.pc_atlas_depth3_pruning_trace_20261103_codex --input le-wm-main/results/pc_atlas_oracle_injection_20261103.json --output NEW_OUTPUT.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main python -m pytest -q -p no:cacheprovider le-wm-main/tests/test_pc_atlas_depth3_pruning_trace_20261103_codex.py le-wm-main/tests/test_pc_atlas_injection_pruning_audit_20261103_codex.py le-wm-main/tests/test_pc_atlas_oracle_injection_20261103.py le-wm-main/tests/test_d2p_atlas_search_v2.py
```
