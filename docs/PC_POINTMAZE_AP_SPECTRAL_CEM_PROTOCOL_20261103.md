# Same-optimizer spectral injection: exposed-root development experiment

Use all fixed exposed roots 1888–1895 and their existing initial and goal
RGB. Both arms use the frozen PointMaze model, predicted-proprio final-goal
score, same action bounds, same zero-raw prior, same two real-RGB feedback
opportunities, same per-root/per-node random seeds and AP-compatible CEM:
16 proposals, 3 iterations, 4 elites, plus prior-versus-mean comparison.
Each decision executes five arbitrary bounded actions. RGB frames 4 and 5
provide causal velocity for the second decision.

On the spectral arm at each decision, compute two JVPs of the predicted
one-packet proprio token with respect to a repeated, raw two-dimensional
action at zero; take the top right singular direction. Replace first
generation candidates 1–4 with repeated raw actions `+.3, -.3, +.6, -.6`
times that direction. Candidate 0 stays the shared zero-raw prior. Normalize
once. Both arms score 50 candidates per decision, but the spectral arm
additionally incurs two JVPs and an SVD, so this is not equal-total-compute.
Use the same modified CEM implementation without injection for the baseline;
verify that its no-injection output agrees with the official-compatible
kernel under a fixed analytic callback and generator.

Execute every arm for all fixed roots, save actions, RGB histories, chosen
plans, singular data, costs and resource counts. Reproduce final RGB and
position independently from saved actions. Existing roots and prior runs
are development data: neither a P1 mechanism nor an independent benchmark
or overall AP reproduction follows. Do not select roots by result.
