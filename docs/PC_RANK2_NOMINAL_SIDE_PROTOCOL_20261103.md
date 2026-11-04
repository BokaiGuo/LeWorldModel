# Rank-two nonzero nominal response: development protocol

Fixed before execution. Preserve previous rank-one experiment and all gates.
Use the existing exact two-action unicycle and 16 exposed radius-.9 goals
at angles 2*pi*(i+.37)/16. This is not RGB or independent confirmation.

Eight nominal turns are linspace(-TURN, TURN, 9) excluding zero; nominal
velocities are (.25, -.25). At each nominal, compute the endpoint and a
2x2 velocity response by central differences with epsilon=1e-5. Record its
rank and both singular values. Compare:

- task: retain one mode by damped task-objective reduction;
- gain: retain one mode by singular-value order;
- ridge: full response solved without SVD;
- random: eight bounded full-action samples, seed 2026111300+root;
- task_greedy: identical task bank, ranked by immediate prefix distance.

For response arms use damping 1e-4, add the solved increment to nominal
velocities, clip velocities to [-1,1], and evaluate the complete chain.
Score all eight candidates, retain four starts, then jointly optimize all
three effective action variables with the existing common bounded L-BFGS-B.
The best of those four scored starts remains an incumbent. Refined prefixes
are allowed to change; evaluate mechanisms on the actual selected chain.

Caps are 64/128/256 charged objective-equivalent calls. Each nominal endpoint,
finite-difference endpoint and candidate score costs one. Greedy prefix
scores also cost one. Random candidates cost one full-chain score each.
Remaining budget is split over four starts, with existing early convergence
behavior and no redistribution. Actual calls, analytic gradients, SVD/solve
counts and elapsed time are separate. This is not a FLOPs fairness claim.
Rank instrumentation for ridge is post-selection and diagnostic only, never
an input to its no-SVD solver. Diagnostic replays are outside planner budget.

For every final selected chain, replay bounds and terminal distance; record
prefix progress and the fixed one-step optimal-suffix/common-selected-suffix
intervention against zero and task_greedy. These limited exact-model probes
are not a fixed multi-suffix family, RGB intervention or P1 confirmation.

Report all roots and budgets, pre/post-refinement quality, paired arm
differences, ranks, costs and mechanism counts. No outcome-dependent
root selection, tuning, gate changes or independent-confirmation claims.
