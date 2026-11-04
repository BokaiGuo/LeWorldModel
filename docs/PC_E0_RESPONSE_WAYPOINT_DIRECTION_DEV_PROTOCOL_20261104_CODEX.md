# Response-derived waypoint direction: E0 development protocol

## Hypothesis and scope

The preceding goal-relative waypoint baseline found two usable chains on an
exposed wall family, but hard-coded a perpendicular geometric direction.
Here test whether an actual local model response can supply a useful
exploration direction, using the same bounded prefix/suffix optimizer and
common random numbers as controls. This is a new 2-D E0 development candidate,
not the existing v11 RGB compiler. Its midpoint, 1.2-goal-distance waypoint
radius, two signs, five/ten-step decomposition and CEM allocations remain
hand-designed and informed by the wall family. It is not a general-purpose
visual feature method or an independent PC2/PC3 confirmation.

At a five-step nominal action of `[0.25, 0]` repeated, evaluate its response
matrix by the same 21 central differences with step 0.05 as v11. Compute an
SVD and discard singular values ≤1e-10. Of the remaining left singular
vectors, choose the one with smallest absolute cosine to the goal direction;
ties follow SVD order. Orient the vector so its first coordinate with absolute
value >1e-12 is positive, then form two target features at the midpoint
between zero-prefix feature and the goal, plus/minus `1.2*||goal-x0||` times
that unit direction. Zero response rank aborts the arm and is recorded.

The matched controls use (a) one fixed-seed isotropic random unit vector and
(b) the old 2-D goal-perpendicular vector. Both also evaluate the same 21
probes and perform the same SVD so the budget and prefix/suffix opportunity
are matched; this is an isolation control, not an efficient deployed random
planner. Each uses the same paired optimizer seeds as the response arm.
Scoring, float32 actions, 160 prefix samples, 220 suffix samples per sign,
full-chain incumbent retention, remaining-budget full refinement, final
zero-first-segment pair, and 1920-token cap otherwise follow the prior waypoint
baseline. Record actual predictor inputs, response, direction, action
proposals, all scores and computational costs. The full-response construction
and SVD costs are not free. Two-dimensional singular directions do not
establish high-dimensional spectral efficiency.

## Fixed development panel and comparison

Eight roots are the Cartesian product of start `[-0.5,y,theta]` with `y` in
`{-0.14,0.16}`, `theta` in `{-0.4,0.75}`, goal `[0.5,gy]` with `gy` in
`{-0.12,0.08}`. Exact task identities must differ from the cited v5, v10
and earlier waypoint panels; broader historical independence is unknown.
Other arms are the same v11 one-mode candidate and three-start direct CEM as
the preceding panel, each under the same 1920-token cap. Use one prespecified
seed per root/arm as set by the runner. Report all results including failures,
final distance and `<0.05` success, candidate quality and actual cost.

All five arms must run once on all roots, without result-conditioned parameter
changes, sample extensions or reinterpretation as confirmation. Recompute every
trajectory, budget, selected chain and response matrix; separately replay with
the independent wall line-intersection implementation. SVD signs and repeated
singular values can be unstable: record the singular spectrum and actual
directions rather than assuming coordinate invariance. Equal packet counts do
not imply equal total computation. No PC gate is passed by this development
test; future PC2 requires the final chain's own prespecified intervention.
