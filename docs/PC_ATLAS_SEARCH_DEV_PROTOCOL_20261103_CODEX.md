# PC2 atlas search development protocol

This is a new development protocol, not a modification of any historical gate.
Use the frozen backend interface and bounded raw-action packets. At every
expanded predicted context, construct zero, one-packet progress, spectral and
seeded random bridge candidates; relinearize after every retained prefix.
Search a fixed-depth beam, ranking completed plans by the bounded nonlinear
model terminal cost, with deterministic tie breaking. Compare the selected
complete plan with the zero-prefix and same-bank greedy-prefix complete plans.
Report no qualified reconfiguration when a non-improving first prefix cannot
strictly beat both controls by the declared margin.

Count actual compiler rollout and Jacobian calls, including direct complete
plan replay; report them separately. This does not certify backend neural
forward/backward counts, encoding, GPU time, or fairness against another
planner. Model replay is not a physical intervention. Development results
cannot establish PC2, PC3, independent generalization, or change legacy gates.
Random seeds, depth, beam width, control horizon, and search budgets must be
recorded before any run. Fail closed on budget exhaustion, invalid numbers,
action-bound violations, and inconsistent split/full replay.

## Fixed analytical smoke panel

Run angular indices 0 through 7 of the previously exposed 64-goal unicycle
panel, angle `2*pi*i/64 + 0.031`; these are development cases. Two one-step
packets, beam width 2, two random bridges per node, seed 42+i, spectral modes
2, prefix radius 1, line scales 0/0.25/0.5/1. Compare terminal position
error with the existing bounded four-start, 32-evaluation analytical gradient
baseline. Position-only task metric, action bounds velocity [-1,1] and turn
[-pi/2,pi/2]. Record both methods' counts by type, without claiming equality
between evaluation types. This is a wiring and negative-control smoke, not
an independent or equal-budget efficacy test.
