# Two-dimensional SVD development result

Protocol: `PC_SVD2_SIDE_PROTOCOL_20261103.md`, fixed before execution.
This side experiment uses all 16 previously exposed exact-unicycle directions,
six arms and objective caps 16/32/64 (288 method-root-budget results). It does
not modify main-thread code, reopen PointMaze, or establish any gate admission.

## What was changed

For each of nine turns, finite-difference the final two-dimensional position
with respect to both first and second velocity. Unlike the previous
single-column response, this permits first-step translation. The response is

    A(turn) = [[1, cos(turn)], [0, sin(turn)]].

It has rank two for the eight nonzero turns and rank one at zero. Both SVD
modes use ridge gains s/(s^2 + 1e-4). The same-matrix rank-one ablation keeps
only its leading mode. A no-SVD control solves the same ridge normal equations.
All three use the same bounded candidates, scoring and full-action optimizer.
This is two-dimensional response SVD, not an image-oriented 2D-SVD algorithm
and not a claim that a single-step velocity/turn Jacobian has rank two.

## Mean final distance across the 16 exposed directions

| Objective cap | Two-mode SVD | One-mode, same matrix | Same ridge, no SVD | Old turn-only | Fixed starts | Random starts |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 7.7485e-5 | 9.8601e-2 | 7.7485e-5 | 1.2308e-1 | 7.3176e-2 | 3.5126e-2 |
| 32 | 2.2626e-8 | 1.6463e-4 | 2.2626e-8 | 7.3778e-7 | 9.2518e-6 | 2.5492e-6 |
| 64 | 2.2626e-8 | 5.6614e-8 | 2.2626e-8 | 8.2413e-8 | 1.1026e-7 | 2.5256e-7 |

The second mode helps this initializer relative to truncating the SAME matrix.
Comparison to the old bank additionally changes prefix coverage and damping,
so that comparison is not a single-variable test of SVD dimension.

The full SVD and no-SVD ridge control agree to numerical precision in all 48
root-budget pairs. That agreement is expected from the identical linear
operator, not evidence of an independent SVD advantage. At turns +/-pi/2,
the two response columns span the two coordinate axes, making the radius-0.9
goals especially simple; these results are not evidence of difficult planning.

Audit detail: an initial test requiring identical ordered starts failed because
roundoff swaps nearly tied symmetric candidates in some roots. No planner,
tie rule, or saved result was changed. The final audit compares candidates at
each fixed angle and resulting terminal errors; it does not claim identical
start ordering, optimizer trajectories, or allocations to each physical start.

## Resources

All response arms score nine generation candidates, consuming nine calls from
their objective cap. Their best scored candidate remains eligible.
The new two-dimensional arms additionally use 36 finite-difference chain
calls (72 dynamics transitions) and 18 candidate-score transitions: 90 in
total. SVD arms use nine decompositions; ridge_no_svd uses nine linear solves.
Old turn-only uses 45 generation transitions and nine SVDs. Fixed/random
perform no structural generation.

The two-mode and ridge arms average 16, 25.375 and 25.375 actual objective
evaluations at caps 16, 32 and 64 respectively. Finite differences are reported
separately, NOT included in the objective cap. Therefore this is not a claim
of equal total dynamics calls, equal FLOPs, or a neural-model efficiency result.
Timing is saved, but this tiny CPU task cannot certify meaningful GPU cost.
Six post-selection diagnostic transitions per result are counted separately.

## Mechanism boundary

At cap 16, none of the two-mode selected chains has non-improving first-step
position error. At caps 32 and 64, two do (roots 3 and 11), in both SVD and
no-SVD ridge arms. Their initial distances are 0.9, prefix distances about
0.9444734, and final distances about 1.7076e-11. They satisfy the recorded
limited same-suffix/optimal-suffix zero-prefix diagnostics. These are actual
selected chains, not substituted candidates.

Those two observations arise after the common full-action optimizer and
occur equally without SVD. They do not establish SVD-specific discovery,
superiority over same-bank greedy, independent mechanism confirmation, RGB
execution, or P1/PC2/PC3 passage. The PointMaze stopped construction remains
stopped.

## Validation and files

Ten tests passed: finite-difference response/rank, full-SVD versus ridge
including rank deficiency, second-mode contribution, six method-specific
budget/bounds/replay checks, and saved-panel source hashes/replay/summary.

- Runner: `le-wm-main/experiments/spectral_reachability/pc_svd2_side_20261103.py`
- Tests: `le-wm-main/tests/test_pc_svd2_side_20261103.py`
- Full results: `le-wm-main/results/pc_svd2_side_20261103.json`

Conclusion: retaining both response dimensions repairs an important limitation
of this initializer, but SVD has no observed incremental benefit over the
equivalent no-SVD response solve. No gates or historical results were edited.
