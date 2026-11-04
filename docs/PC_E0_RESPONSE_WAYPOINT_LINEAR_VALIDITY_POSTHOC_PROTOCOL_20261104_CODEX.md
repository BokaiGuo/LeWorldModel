# Post-hoc response linearity on actual waypoint prefixes

Use the sealed eight-root response/random/geometric waypoint direction run.
For each of its three arms and both saved prefix optimizers, compare the actual
five-step endpoint of the optimized prefix to the linear prediction
`f(nominal) + J @ (prefix - nominal)`, where `J` is that arm's saved central
difference matrix and `nominal` is the fixed repeated `[0.25,0]` action.
Also compare changes in goal distance and record absolute error, displacement
norm, relative error only when displacement is above 1e-10, and sign agreement
of nonzero actual and predicted goal-cost changes. Report all 48 comparisons,
with arm/root/branch identities, rather than selecting successful roots.

Determine whether the actual prefix and nominal take identical wall-blocking
event sequences. This is a post-hoc description of already selected
optimization outputs, not an independent test, a finite-difference accuracy
certificate, an exhaustive local validity radius, or a mechanism test.
Recompute endpoints using the existing exact wall backend and compare with a
separate line-intersection implementation. Save input and implementation
hashes and write output exclusively; no extra optimizer runs or selection
based on this analysis.
