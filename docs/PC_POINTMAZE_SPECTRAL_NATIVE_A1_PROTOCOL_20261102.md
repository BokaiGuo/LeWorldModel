# Development A1 probe: model Jacobian direction with native RGB execution

Fixed before the run: exposed archive roots 1880–1883; frame 0 reset; frame
10 RGB goal; frozen PointMaze checkpoint; initial native RGB rendered by the
development-selected camera. The model's unavailable proprio input is the
fixed raw zero vector. Full-archive normalization is inherited unchanged,
so its training-only provenance is unknown.

A candidate is a constant two-dimensional physical action repeated for five
archive actions. At zero action, differentiate the model's five-action
pooled visual successor with respect to the two physical action components,
including exactly one model-action normalization. Take the principal right
singular vector of this 384-by-2 Jacobian. Evaluate these fixed five-action
candidate sets with the same frozen model and the same terminal goal-latent
distance:

- `spectral`: zero, plus/minus 0.3 and plus/minus 0.6 along that unit vector.
- `random`: zero, plus/minus 0.3 and plus/minus 0.6 along a unit direction
  drawn from NumPy's fixed seed `20261102 + root`.
- `axis`: zero and plus/minus 0.6 along each physical coordinate.

Each method selects the minimum model cost, executes its selected five-step
action through a separate native RGB environment, and records the resulting
RGB hash. Compare the offline position-only replay distance against the
frame-10 position only after all three model selections. The goal's later
timestamp is deliberate and common; the experiment measures the result
after five steps and is not a success-rate metric. Retain all three methods
for all four roots, including ties or failures.

Record Jacobian/SVD calculation, finite-difference validation, candidate
evaluation and wall time separately. The spectral method incurs additional
derivative computation, so identical five-candidate forward scoring does
**not** mean equal total computation. Fixed random direction is a weak
matched-dimensional diagnostic, not CEM/GRASP or a strong baseline. Do not
claim the full task-relevant Q-weighted D2P reconfiguration mechanism from
an unweighted one-packet SVD. These are development roots, not independent
confirmation, and no gate is modified.
