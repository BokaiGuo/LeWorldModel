# Prefix/tail exchange and normalized-step results

Two prospective four-arm implementations were executed on host CUDA. All eight
trials are retained. Unchanged native evaluator, checkpoint and P1 gate.

| Arm | Exchange: success / jointXY at20 | Normalized exchange: success / jointXY at20 |
| --- | --- | --- |
| structured | no /31.09926 | no /79.65458 |
| native | no /27.79804 | no /36.82156 |
| random | no /64.43621 | no /56.73130 |
| immediate_progress | no /36.56573 | yes at16 /unavailable at20 |

Early native success is not padded to step20. The full16-slot historical/current
comparison is `le-wm-main/results/p1_search_revision_analysis_20261101_main_MAIN02.json`.
Its checkpoint list is distinct from the native stopping step.

## Exchange implementation

All64 prefix/tail combinations were scored at b0/b5; no redundant H1 exchange.
All four initial winners remained their original diagonal combinations.
At b5, three non-structured arms found better off-diagonal model scores.
Structured did not change its winner at either exchange node.
All16 nodes passed diagonal equality, source reconstruction, argmin and cost checks.
Added320 predicted packets per full trajectory were charged, not hidden.

## Normalized action-space update

Same neural call counts as exchange for the corresponding arm. Six updates per
candidate; each gradient divided by its infinity norm, retaining alpha=.05.
All16 decision nodes passed source/selection/cost checks and eight×six update-log
checks per node. Structured's actual peak updates were approximately.05 throughout.
Its initial choice changed to the zero prefix, while the immediate-progress
baseline succeeded at16. Thus small updates were a useful optimizer hypothesis,
but larger updates do NOT establish a D2P advantage or reconfiguration mechanism.

Both new implementations are retained as experimental variants. They do not
overwrite the prior Q-winner implementation, which had one structured success
on this known task but no qualifying non-greedy prefix. No cherry-picked winner
is declared the final method, and no independent or external-baseline claim follows.

Next protocol-level investigation: a separately specified opposite-side push
task with an actual20-step witness. A scripted witness is task eligibility only;
the RGB planner must independently generate a successful mechanism.
