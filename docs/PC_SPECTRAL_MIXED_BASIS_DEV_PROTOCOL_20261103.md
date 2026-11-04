# Mixed-action spectral candidate control

Development-only follow-up to the exposed eight-goal unicycle ablation. The
previous spectral and coordinate arms only generated single-axis candidates,
whereas the random arm generated simultaneous velocity and turn. This
protocol compares mixed actions under the same candidate count and the same
bounded nonlinear suffix optimizer; no extra model training or physical
claim is involved.

Use goals `2*pi*i/64+0.031`, i=0..7, and four predetermined random seeds
101, 202, 303 and 404, all development cases. Each arm uses 16 proposals plus
the same zero and one-step greedy control. Generate 16 coefficient vectors
uniformly in the 2-D unit disk using the same seed for all arms in each
case. Arms:

1. `raw_isotropic`: use disk vectors directly as raw actions.
2. `spectral_isotropic`: rotate the same vectors through the full task-weighted
   right singular basis at the root, including the null direction.
3. `spectral_anisotropic`: as (2), but multiply the coefficient of the
   second singular direction by 0.25 before rotation.

The disk and action scaling are deliberately identical and inside the raw
velocity and turn limits. The first two distributions are mathematically
identical under an orthogonal basis change, so differences are Monte Carlo
variation rather than evidence of a useful geometry. The third intentionally
changes the proposal distribution; it tests a specific heuristic, not a
pure basis-only effect. The coefficient order is descending singular value;
no goal-specific tuning or physical outcome enters proposal construction.

For every prefix, recompute the same `LocalCompiler.progress` suffix with one
packet, the same damping and line-search rules, then replay the full two-action
chain and select by nonlinear terminal task metric. Record all actions,
costs, attempted/returned rollout and Jacobian calls, source code SHA, seed,
selected index and per-arm wall time. Spectral arms pay for their additional
Jacobian/SVD. Counts and timing are reported separately; equal candidates do
not establish equal computational budget. Root outcomes, seeds and candidate
counts were selected before this new run, but the roots are exposed and this
is not a confirmatory experiment.
