# Fixed-bank candidate-source ablation, analytical development

Purpose: distinguish structured candidate generation from ordinary two-stage
lookahead, not claim final physical planning performance. Reuse exactly the
eight exposed angular unicycle goals and two-step horizon in the oracle
injection diagnostic. Three separately instantiated arms use four candidate
prefixes plus identical zero and one-step greedy controls, and the same
`LocalCompiler.progress` suffix optimizer at every predicted successor.
Every candidate receives one suffix optimization and full root replay.

Candidate sources are: (a) two leading right-singular response directions
and their negatives; (b) four seeded uniformly sampled action prefixes;
(c) four coordinate-axis ± unit prefixes, a deterministic ordinary-lookahead
control. Spectral uses the full task-weighted right basis, including zero
singular directions, with full orthonormal basis made available when needed.
All actions are in the same raw domain. Use fixed seed 100+i for random,
fixed radius 1, same one-packet suffix damping and line-search configuration,
and select minimum full nonlinear terminal metric. Record all candidate and
control actions/costs, rollout/Jacobian calls by arm and wall time. SVD occurs
only in the spectral arm, and its Jacobian/rollout calls are explicitly
charged; common greedy and suffix construction also use SVD internally in all
arms. Equal candidate counts and suffix policy do not imply equal compute.

This analysis uses known analytic dynamics, existing development goals,
different candidate generation costs, and a tiny panel. Results are
exploratory; no PC or historical gate changes, no independent performance
claim, and no real-RGB or frozen-model inference.
