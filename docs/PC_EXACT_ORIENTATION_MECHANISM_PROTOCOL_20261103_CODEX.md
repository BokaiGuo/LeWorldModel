# Exact orientation mechanism: development protocol

This is a new, synthetic exact-dynamics development experiment, not an RGB
planner, independent confirmation, or an admission test. It does not reopen
the stopped PointMaze non-immediate mechanism construction.

## Task fixed from dynamics

Use the existing bounded two-step unicycle: translation at each step is
along the current heading; the turn changes the next heading. A turn with
zero translation preserves position but changes the direction of the next
translation response. Starting at `(0, 0, 0)`, evaluate 16 unit-direction
goals of radius 0.9 at angles `2*pi*(i + 0.37)/16`. This panel is specified
geometrically, not selected from previous outcomes.

## Questions and comparisons

1. Existence diagnostic: enumerate nine evenly spaced turn-only prefixes in
   `[-pi/2, pi/2]`. At each successor compute the translation response by
   centered finite differences, extract its one-dimensional action singular
   direction using SVD, and optimize bounded forward/backward translation by
   projection onto that response. Select the best full two-step chain by final
   squared distance. This restricted diagnostic deliberately enforces
   position-preserving prefixes; it is not evidence that an unrestricted
   planner selects them.
2. Compare the selected prefix and zero prefix using the same selected
   translation suffix, recording each absolute final distance and each
   improvement relative to its own zero suffix. Also compare their individually
   optimized translation suffixes.
3. Run the existing four-start full-action bounded L-BFGS-B optimizer at
   budgets 8, 16 and 32 objective/analytic-gradient evaluations and four
   independent, fixed initial seeds for full-action CEM, each at 32 candidate
   evaluations. CEM uses four batches of eight and two elites. Report costs,
   full actions, objective/gradient calls, dynamics transitions, SVD counts
   and wall time separately. Do not claim the operation categories are equal.
4. A combined unrestricted selector chooses the lower terminal error of
   the structured chain and the 32-evaluation gradient chain, ties favoring
   the latter. Its cost is the sum of both procedures. Its selected chain,
   not the restricted diagnostic chain, determines whether the combined
   method actually selected non-immediate progress.

Report all 16 rows, the restricted witness count, full-action baseline wins,
and combined-selector mechanism count. The existence of a restricted witness
does not imply search efficiency. If the unrestricted selector does not select
a useful non-immediate chain or full-action search already solves the task,
do not claim a spectral search advantage; use this as an exact-task limitation.
No outcomes flow to the visual planner. Fail on preexisting output.
