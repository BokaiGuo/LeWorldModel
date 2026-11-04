# PointMaze five-action task-response bank: development protocol

Use all four previously exposed obstacle roots 8343, 8345, 8349 and 8351
with the frozen checkpoint, existing RGB state estimator and analytic affine
inverse. This is exploratory development, not an independent P1/PC test.

At each initial RGB observation, compute the frozen model's one-packet
successor predicted proprio representation as a function of the full five
raw two-dimensional actions (10 coordinates). Map its mean representation
through the existing affine inverse and position scales. Compute its 2 × 10
Jacobian at the all-zero raw action using ten JVPs. Independently check
each column with centered finite differences of step 0.001. Save both
Jacobians, absolute and relative errors and singular values; record, rather
than silently omit, numerical disagreement.

Construct the fixed bank of nine bounded full five-step programs:
zero, ±0.5 in infinity norm along each of the first two right singular
directions, and ±0.5 along each of the first two SVD nullspace basis
directions. Require both nonzero modes to exceed a fixed relative tolerance
of 1e-5 and 8-dimensional nullspace; otherwise fail closed and report the
actual rank. Basis vectors in a multidimensional nullspace are not unique;
save actual vectors and actions, and do not claim rotational invariance.
No physical state, terminal outcome or privileged velocity participates
in candidate generation or model scoring.

Before physical outcomes are loaded, score all nine model predictions by the
existing affine position proxy. Then execute every fixed bank member from the
same root for five real simulator actions, retaining all images and actions.
Check reset pixels against the sealed source; encode each actual cp5 image
with causal cp4/cp5 RGB velocity, and separately evaluate physical endpoint
using the existing position backend. Report per-root within-bank pairwise
rank reversals for predicted score versus actual RGB-derived encoded score
and physical goal distance; preserve all raw numerical outputs and exact
candidate order. Compare the finite nonlinear task response with its linear
prediction at the actual action magnitude, not only infinitesimal FD.

Account separately for encodes, JVP calls, finite-difference samples,
candidate predictions, RGB physical steps, position-backend replay steps and
wall time; bind input and output hashes and verify frozen parameters before
and after. Initial velocity uses the existing single-frame zero assumption;
the cp5 RGB velocity is a temporal average. The affine representation is
derived from RGB-estimated state, not an unmodified visual-latent objective.
No successor optimization, common-suffix intervention, equal-total-compute
comparison, new-root confirmation or mechanism admission is implied.
