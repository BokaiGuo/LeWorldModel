# Spectral initialization: exact-dynamics diagnostic protocol

This development-only diagnostic uses the four gradient-selected witness
chains already identified in the sealed 16-goal exact-orientation experiment.
The four roots were selected after their outcomes were observed. No RGB method,
independent generalization, or gate is evaluated.

## Known-plan injection

For each fixed root, replay the sealed gradient-selected chain and enumerate
the nine existing turn-only spectral candidates. Measure prefix action
coverage, including both first velocity and turn, and score the injected chain
through the same exact full-chain replay used for the nine candidates. Rank
all ten with stable ties. Injection is diagnostic only and is never used as a
method initialization or planning score. Record that the existing restricted
bank does not have an intermediate beam-pruning step; do not infer what a
different Atlas implementation would prune. Separately fix the gradient
prefix and optimize its one-step suffix analytically to assess the suffix
objective, and compare its result with the replayed gradient suffix.

## Common-optimizer initialization comparison

On those same four fixed roots, use identical bounded full-action L-BFGS-B
and its exact objective/analytic gradient. Four starts per method:

* `spectral`: compute nine finite-difference successor response matrices and
  SVDs, construct turn-only projection chains, score all nine full chains,
  and use the four lowest-scoring chains as initial points.
* `response_no_svd`: same angles, finite-difference response, projection,
  scores and four-start selection, without an SVD. This isolates whether the
  one-dimensional SVD adds anything beyond the response projection.
* `fixed`: the four generic starts of the original gradient baseline.
* `random`: four uniform bounded starts, seed `2026110800 + root_index`.

The objective-evaluation caps are 16, 32 and 64 per method. Candidate
generation scores for the two response methods consume nine evaluations from
the same cap; gradient objective/gradient calls consume the remainder. Split
the remaining evaluations equally over the four starts in stable order;
if a start converges early, do not reassign its unused allocation. Report
actual objective calls, analytical gradients, finite-difference transitions,
SVDs, wall time and the terminal distance separately. These heterogeneous
operations are not equated to FLOPs, and a common objective cap plus reported
structural overhead is not full resource fairness. Because scoring the initial
programs is part of search, the best initial program also remains eligible as
the final answer. Record results at every prespecified budget; no post-hoc
choice of budget or root supports a performance claim.

The four witness chains are injected only into the diagnostic in the first
section, not into any method. Differences between `spectral` and
`response_no_svd` are evaluated on identical numerical inputs; report exact
equality or floating-point deviations. These tiny, selected development roots
are suitable for locating a search gap, not for freezing or independent
confirmation. Keep all outcomes and stopping decisions separate from the
previous PointMaze non-immediate mechanism.
