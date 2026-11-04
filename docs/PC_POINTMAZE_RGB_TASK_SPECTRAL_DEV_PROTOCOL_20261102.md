# RGB task-region weighted spectral action development

On the four already exposed development roots 1884–1887, compare three
five-slot prefix banks: top right singular direction of the frozen model's
full spatial action Jacobian; top direction of the same Jacobian restricted
to RGB task-region tokens; and a seeded random direction. The Jacobian maps
one bounded, constant five-action packet to predicted spatial visual tokens,
evaluated at raw zero action, with two physical-action JVPs. Each bank uses
zero and signed magnitudes 0.3 and 0.6. These are one-layer progress
candidates, not a reconfiguration or full D2P program.

Task-region tokens come solely from current and goal RGB. Apply the existing
PointMaze green-marker color rule, reject missing markers, masks with fewer
than 4 or more than 500 pixels, markers with bounding boxes wider/taller than
20 pixels, or centroids within 14 pixels of image borders. Map each centroid
to its 14-by-14-pixel DINO token and include that token plus the surrounding
one-token neighborhood; take the union for current and goal. The weights are
binary and normalized by the number of selected tokens. This is an
environment-specific RGB heuristic, not learned task geometry.

Within each bank select the lowest nonlinear model rollout MSE over these
tokens; execute five actions, observe native RGB, recompute the task-region
mask using the actual feedback and same goal, then choose and execute one of
the previously fixed eight five-action tails with the same token-MSE scoring.
Use fixed-zero raw proprio; no simulator state, future observation or outcome
enters any decision. Evaluate actual position distance only after all three
arms decide. Same-prefix RGB must reproduce; save all prefix/tail arrays,
scores, masks, Jacobian columns, images, resource counts and helper hashes.

The task-weighted and full-spatial spectral families share the same two
JVPs but each incurs an SVD; random does not. All methods score five prefixes
and eight tails, so candidate count equality does not imply total-compute
equality. Encoding and model weights stay frozen; all neural work is on host
CUDA. These roots have been used for method development, checkpoint training
membership is unknown, normalization uses the full archive and the camera
was chosen on an exposed root. The metric is position-only offline replay,
not native success. This is not independent confirmation, complete RGB-only
model authentication, or P1.
