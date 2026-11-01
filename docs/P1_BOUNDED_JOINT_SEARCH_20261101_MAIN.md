# Model-only bounded joint action search

An independently named development run optimized three five-action packets
from eight existing spectral8 candidates per known root. It fixed eight
projected gradient updates per initial candidate and the objective
`q3 + 10 * relu(q0 - q1)^2` before the run. A candidate was selectable only
when its first-packet latent MSE was at least the root value and its
three-packet MSE was strictly lower.

| Root | Root MSE | Iterates satisfying q1 >= q0 | Minimum q3 among those iterates |
|---|---:|---:|---:|
| 0 | 1.421916 | 41 / 72 | 1.544188 |
| 1 | 0.378106 | 13 / 72 | 0.616895 |

No iterate satisfied both selection conditions. The model run recorded 432
action-encoder calls, 432 predictor calls and 128 reverse requests, with
matching returns and module-hook counts, on host CUDA. Frozen model hashes
before and after matched. The saved-record verifier does not rerun inference.
Historical candidate generation, RGB encoding and system GPU costs are not
included. No physical execution was authorized by this model-only protocol.

This finite search on already-used roots does not prove physical impossibility,
model correctness or a theoretical limit; it also does not admit P1. The
result is at
`le-wm-main/results/codex_p1_bounded_joint_20261101_exclusive_RUN_CODEXJOINT01/report.json`,
with code and the separate saved-record verifier in
`le-wm-main/experiments/spectral_reachability/`.

## Fixed 32-update follow-up

A separately sealed 32-update development run retained the same eight starts,
learning rate, penalty, model and hard selection rule. Its first nine iterates
match the earlier run. At root0, candidate 4 at iteration 13 was the only
feasible iterate among 264: q0=1.421916, q1=1.432043 and q3=0.453772.
No root1 iterate was feasible. The next root0 iterate violated the q1
constraint, so this is not evidence of convergence or robust feasibility.
There was no action projection clipping in this run.

The follow-up recorded 1,584 predictor and action-encoder calls and 512
reverse requests; historical candidate generation and RGB encoding remain
outside its accounting. This is a model-selected plan, not a physical
mechanism result. Physical replay requires separately certified source,
action, RGB and native termination bindings.

## Fixed-plan physical development replay

The sole model-feasible root0 plan was sealed and physically dispatched
without changing its actions using the physical outcome. The frozen model,
goal, raw action domain, initial physical state and RGB were bound to the
source; saved action receipts and all 15 returned actions were independently
checked.

| Control step | Pusher XY | Object XY | Wrapped object angle | Joint XY |
|---|---:|---:|---:|---:|
| 0 | 141.821 | 101.676 | 0.6222 | 174.503 |
| 5 | 143.084 | 101.676 | 0.6222 | 175.531 |
| 10 | 81.446 | 66.315 | 0.4125 | 105.029 |
| 15 | 10.448 | 12.973 | 0.0297 | 16.657 |

The fifth-step values show no immediate improvement in these reported
physical metrics. At step 15 both native and strictly wrapped success were
true. Native termination was returned at that step; the runner reports
`full_horizon=false` and `endpoint=null` because it handles termination before
loop completion, while the saved verifier confirms that all 15 actions and
the final observation were returned. This is a reporting-semantic boundary,
not an unobserved final state.

This is an exploratory fixed action chain on one known development root.
It does not yet establish a causal ability change, independently planned
suffix superiority, total-budget fairness or independent generalization.
The historical candidate generation and RGB costs are not accounted in the
model-only totals above. Reports:
`le-wm-main/results/codex_p1_diag32_fixed_physical_20261101_exclusive_run_CODEXPHYS01/report.json`
and
`le-wm-main/results/codex_p1_diag32_fixed_physical_verify_20261101_exclusive_CODEXPHYS01.json`.

## Same-bank common-tail development intervention

After that outcome was observed, six further fixed physical branches used
the selected, zero and model-greedy prefixes, each followed by the selected
plan's same ten-action tail or a ten-action zero tail. At step 15, selected
plus common tail reached joint XY 16.657 and native success, versus 176.162
with zero tail; zero prefix plus common tail reached 21.106 without native
success, versus 175.177 with zero tail. The same-bank greedy/common branch
reached native success at step 10 and was stopped; no step-15 value exists.

These results show action- and prefix-dependent consequences for this fixed
tail, not an optimized-suffix capacity comparison or independent
confirmation. They do not establish superiority over greedy. The greedy
selection is the first minimum first-packet model cost over all 264 saved
root0 iterates. Full receipt and early-stop details are in
`docs/D2P_P1_DIAG32_SAMEBANK_TAIL_RESULT_20261101_CODEX.md`.

## Per-prefix real-RGB suffix optimization

The selected, zero and same-bank greedy prefixes were separately replayed
from the same root. At each actual cp5, the branch's own RGB and executed
action history were used by the same optimizer and fixed suffix budget; its
selected tail was then physically dispatched in that same episode.

| Prefix | cp5 joint XY | Last observed step | Last joint XY | Native and wrapped success |
|---|---:|---:|---:|---|
| selected | 175.531 | 15 | 35.118 | no |
| zero | 175.177 | 15 | 33.563 | no |
| same-bank greedy | 96.473 | 14 | 17.390 | yes |

The greedy branch stopped at step 14 and its final unexecuted action was not
padded. Each branch used one cp5 RGB encoding, 528 predictor calls,
528 action-encoder calls and 256 reverse requests; module hooks matched the
attempt and return records. Historical prefix construction and RGB costs are
not balanced, so this is not certified equal end-to-end compute. The single
known-root result does not support superior selected-prefix plannability:
zero obtained a slightly smaller final joint distance, while greedy reached
success. This is a development observation, not a claim of global failure.

The run and saved-artifact verifier are
`le-wm-main/results/codex_p1_diag32_cp5_live_20261101_exclusive_run_CODEXLIVE01/report.json`
and
`le-wm-main/results/codex_p1_diag32_cp5_live_verify_20261101_exclusive_CODEXLIVE01.json`.
The verifier checks recorded identities and binding; it does not independently
rerun neural inference or physics.

A subsequent CUDA wiring spot check re-encoded all three cp5 RGB frames,
recomputed one first optimizer update per arm and each recorded winning
score, with maximum absolute difference zero in these checked quantities.
It also verified the actual five-action prefix in the normalized history and
the common goal binding. It did not rerun all gradients and cannot exclude
every code error. The cp5 optimizer scores its new two-packet model latent
MSE; it did not carry the original successful selected tail as a candidate.
Accordingly, the replacement of that tail is a concrete method difference,
not proof that the model or its theoretical mechanism is impossible.

## Same-action prediction versus real RGB encoding

The four observed frames were encoded on host CUDA and compared with an
open-loop model rollout of the same sealed actions:

| Step | Predicted goal MSE | Real RGB latent goal MSE | Predicted/real latent RMSE |
|---|---:|---:|---:|
| 0 | 1.4219 | 1.4219 | 0.0000 |
| 5 | 1.4320 | 1.4083 | 0.1020 |
| 10 | 0.8787 | 0.8093 | 0.3247 |
| 15 | 0.4538 | 0.0484 | 0.6072 |

Thus the first-packet goal-MSE change has opposite signs in prediction and
real RGB encoding, even though the physical object pose did not improve.
The last column is latent error, not pixel error. A single shared trajectory
cannot quantify generalization. The diagnostic and its actual CUDA calls are
separately sealed in
`le-wm-main/results/p1_diag32_same_action_rgb_20261101_leibniz_RUN_LEIBNIZ01/report.json`;
the ledger remains `closed_unverified`.
