# Two-dimensional response SVD: bounded side experiment

This protocol is fixed before running the new experiment. It improves the
existing exact unicycle initialization diagnostic only. It does not reopen
PointMaze, alter gates, certify RGB control, or provide independent evidence.
All 16 directions of the already exposed orientation panel are used, not the
four outcome-selected witnesses. Goals are reconstructed from the fixed rule
`0.9 * (cos(a), sin(a)), a = 2*pi*(i+0.37)/16`.

## Meaning of two dimensions

For each of nine fixed turns in [-pi/2, pi/2], use the two-step program
`[(v0, turn), (v1, 0)]`. Estimate the 2x2 map from `(v0,v1)` to final `(x,y)`
using central differences at zero, epsilon 1e-5. This removes the old
first-velocity-zero restriction. A single-step velocity/turn position
Jacobian would still be rank one in this environment.

At turn zero, keep the actual rank deficiency; never invent a second mode.
Use ridge lambda 1e-4, raw velocity coordinates, and clip both velocities to
[-1,1]. Clipping is a candidate construction, not an exact bounded solve.
The second action's turn remains zero because it cannot affect the terminal
position under these dynamics. The full optimizer can change v0, turn, v1.

## Six arms

1. `svd2`: retain both singular modes of the 2x2 response, using ridge gains
   s/(s*s+lambda).
2. `svd1_same_matrix`: retain only its leading mode; all other settings match.
3. `ridge_no_svd`: solve (A.T A + lambda I)v = A.T goal on the same matrix.
   Full SVD ridge and this solve represent the same linear operator; their
   agreement is expected, not evidence of an independent SVD advantage.
4. `old_turn_only`: reproduce the previous single-column SVD candidate bank.
5. `fixed`: four generic original gradient starts.
6. `random`: four bounded starts, seed 2026110900 + root index.

Each response arm scores nine complete candidates and sends its best four
to the same bounded full-action L-BFGS-B optimizer. The best already scored
candidate remains eligible, even if optimization worsens it. Objective caps
are 16, 32, 64, including the nine generation scores. Remaining calls are
split equally in stable order; early-convergence leftovers are not reused.
Fixed/random starts use the whole cap. Finite differences, replay transitions,
SVD/linear solves, analytic gradients and time are reported separately: this
is not an equal-FLOP or end-to-end cost claim.

## Checks and interpretation

Verify finite differences against A=[[1,cos(turn)],[0,sin(turn)]], rank 1 at
zero and rank 2 at nonzero turns; compare the two full-rank solvers; replay
every chosen chain, enforce bounds and caps, retain all roots and budgets.
Report immediate progress and same-selected-chain response diagnostics versus
zero prefix using its same suffix and its exact best bounded suffix. These
are limited exact-model witnesses, not a full mechanism test against greedy
or independent confirmation. Diagnostic replays are excluded and counted
separately from planning.

No new optimization tuning, extra roots, or threshold changes after outcomes.
Stop after this panel and report both gains and null results. Existing source,
result and gate files remain unchanged. New artifacts use the `pc_svd2_side`
prefix and refuse to overwrite an existing result.
