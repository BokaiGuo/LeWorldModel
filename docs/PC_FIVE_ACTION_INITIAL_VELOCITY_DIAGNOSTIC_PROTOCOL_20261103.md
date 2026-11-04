# Fixed five-action initial velocity diagnostic

Reuse all 36 already exposed five-action bank programs, initial and cp5 RGB
frames, and physical endpoints from the bound response-bank receipt. No new
candidate generation, optimizer, action selection, or simulator trajectory
is performed. For each root encode the same initial RGB and RGB-estimated
position with (1) the historical zero velocity and (2) the reset's
instantaneous simulator qvel. The second input is privileged and must never
be passed into the RGB-only planner; it is strictly an offline attribution
diagnostic.

Run the same frozen model on every identical action packet, scoring predicted
proprio via the same affine position proxy. Re-encode the saved cp5 RGB with
the same cp4/cp5 causal RGB velocity used by the bank experiment. Verify the
zero-input predictions and cp5 scores against the original receipt. Only
after prediction and consistency checks, load the already saved physical
position endpoints and report per-root within-bank ranking reversals and
per-candidate errors relative to both actual RGB-derived scores and physical
distance. Save numerical results, source and model hashes, actual callback
counts, and wall time. Simulator qvel is instantaneous while RGB velocities
are temporal averages; observation/render timing is not certified. A
reduction in error is not a deployable-control or closed-loop success claim.
