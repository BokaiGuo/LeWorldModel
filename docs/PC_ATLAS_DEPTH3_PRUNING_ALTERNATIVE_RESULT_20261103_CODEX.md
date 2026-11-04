# Atlas depth-3 non-oracle pruning alternative: development results

## Design and checks

Following the separately written protocol, the new standalone selector
reuses the existing ordinary v2 candidate bank and exact-unicycle backend; it
never reads the saved injected/gradient chains. Only depth-2 branch pruning
changes. The new `lookahead_diverse` mode retains the best one-step
model-lookahead node and, within the fixed 0.01 model-cost tolerance,
reserves a second node with maximally different normalized second action.
The tolerance is an explicit selector option. Ordinary `lookahead` retains
the best two lookahead scores; `v2` uses current node cost; `random` chooses
two nodes using a separate seeded generator.

All arms use depth 3, branch width 2, original per-root-branch search and
terminal model-cost selection. The eight already exposed analytical goals
are each run with three seeds, giving 24 paired runs but only eight distinct
goals. For each pair, an independent run of the actual original `AtlasSearch`
matched the v2 arm's branch actions, prefix/terminal costs, output-context
hashes, selected index and relevant selection indices. This is output parity
on these runs, not a proof for arbitrary backends.

## Results

| Intermediate policy | Mean terminal position error | Mean terminal model cost |
| --- | ---: | ---: |
| Original v2 policy | 0.145677 | 0.010779 |
| Ordinary lookahead | 0.067801 | 0.002750 |
| Lookahead + diversity | 0.067801 | 0.002750 |
| Random retention | 0.144449 | 0.011935 |

Against v2, lookahead + diversity has 15 wins, 9 ties and 0 losses across the
24 paired goal/seed cases (ties use absolute position-error difference
`<= 1e-12`). Its comparison to random is also 15/9/0. However, it has **the
same final selected actions as ordinary lookahead in all 24 cases**. The two
policies selected different ordered intermediate beams in 55 of 144 branch
instances, but those differences did not produce a different final action
chain in this sample. Therefore the observed improvement against these v2 and
random controls cannot be attributed to the added diversity rule; ordinary
lookahead alone is sufficient to explain the observed final choices.

## Matched-call accounting

Every arm executes the one-step lookahead for every depth-2 candidate.
The v2 and random arms discard it: this deliberate padding equalizes backend
calls but is **not** their natural compute cost. Per arm and paired run:

- 340 rollout attempts/returns, covering 352 rollout action steps;
- 55 Jacobian attempts/returns, covering 55 Jacobian action steps.

All four quantities matched exactly in each pair. Across the 96 experiment
arms this is 32,640 rollout calls, 33,792 rollout action steps, 5,280 Jacobian
calls and 5,280 Jacobian action steps, with all attempts returned. Separate
original-v2 parity checks used another 4,704 rollout and 456 Jacobian calls
across 24 runs. An unpadded original run used 196 rollouts, 208 rollout
action steps, and 19 one-step Jacobians in this experiment. Tests and
preflight are excluded. Runtime, allocation, extra distance calculations,
trace storage and total hardware cost are not matched.

The standalone selector allows padding to be disabled; lookahead modes still
perform their necessary trial calls. It is limited to the implemented depth 3
and beam width 2, and this experiment uses one-step packets. No evidence here
establishes equal runtime, performance on learned visual models, physical
control, or generalization. The 0.01 tolerance and action-space distance are
development choices, not validated general hyperparameters. This is not
independent confirmation or an admission gate.

## Artifacts

- Pre-run protocol: `docs/PC_ATLAS_DEPTH3_PRUNING_ALTERNATIVE_PROTOCOL_20261103_CODEX.md`
- Independent selector and runner: `le-wm-main/experiments/spectral_reachability/pc_atlas_depth3_pruning_alternative_20261103_codex.py`
- Detailed candidate traces, selected actions, hashes and counts: `le-wm-main/results/pc_atlas_depth3_pruning_alternative_20261103_codex.json`
- Tests: `le-wm-main/tests/test_pc_atlas_depth3_pruning_alternative_20261103_codex.py`
- Separate decision-reason audit: `le-wm-main/experiments/spectral_reachability/pc_atlas_depth3_pruning_decision_audit_20261103_codex.py`
- Per-candidate ranks, eligibility, distance and reason: `le-wm-main/results/pc_atlas_depth3_pruning_decision_audit_20261103_codex.json`
- Decision audit tests: `le-wm-main/tests/test_pc_atlas_depth3_pruning_decision_audit_20261103_codex.py`

The new tests and relevant old search/depth-3 tests passed: **20 passed**.
The existing code, protocols and results were not altered.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main python -m experiments.spectral_reachability.pc_atlas_depth3_pruning_alternative_20261103_codex --output NEW_OUTPUT.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main python -m experiments.spectral_reachability.pc_atlas_depth3_pruning_decision_audit_20261103_codex --input NEW_OUTPUT.json --output NEW_AUDIT.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main python -m pytest -q -p no:cacheprovider le-wm-main/tests/test_pc_atlas_depth3_pruning_decision_audit_20261103_codex.py le-wm-main/tests/test_pc_atlas_depth3_pruning_alternative_20261103_codex.py le-wm-main/tests/test_d2p_atlas_search_v2.py le-wm-main/tests/test_pc_atlas_depth3_pruning_trace_20261103_codex.py
```
