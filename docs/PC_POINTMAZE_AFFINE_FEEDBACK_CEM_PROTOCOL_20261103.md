# Affine-inverse score in real RGB feedback CEM: development protocol

This new development experiment follows the fixed eight exposed roots
1888–1895 and the innovations from
`PC_POINTMAZE_PROPRIO_SCORE_FEEDBACK_CEM_PROTOCOL_20261103.md`. It compares
an untrained affine-inverse scoring arm to its already recorded visual and
proprio arms. Roots and prior outcomes are exposed, so this is not an
independent confirmation or a P1/PC gate test.

Use the frozen PointMaze checkpoint, identical initial and tail CEM
innovations, candidate counts, action scaling, five-step packet clock,
feedback opportunities and horizon from that protocol. Initial state uses
RGB-estimated position with zero velocity; after the arm's own five-step
prefix, derive state from its own real cp4/cp5 RGB and causal velocity,
encode again and optimize the tail. Do not use future candidate-specific
physical observations or true velocity in planning. State is permitted only
for environment reset and offline endpoint evaluation. Record the scene/camera
calibration and the whole-archive normalizer as provenance limitations.

At every candidate batch, use the model's predicted proprio token mean and
the encoded target token. Score the norm of the first two coordinates of
their difference projected by the Moore–Penrose inverse of the fixed
checkpoint encoder, multiplied by the same position normalizer standard
deviation. Fail closed on shape, nonfinite data, nonpositive normalizer,
rank deficiency, checkpoint/normalizer mismatch or model drift. No trained
decoder, policy, critic, simulator state or oracle successor enters scoring.

Execute all eight roots without filtering or replacing failures. Save full
candidate/score traces, actual RGB histories, actions, inputs, timings,
source/model hashes, prediction sample counts, and replay evidence. Compare
per-root endpoint distances with both saved arms, with failures explicit.
Report separately the actual model calls, encodings, elapsed time and
scoring overhead; identical samples do not certify equal total compute.
The arm-specific CEM updates naturally yield different candidate values
despite common innovations. Preserve all results, including negative results.

This test does not establish spectral contribution, non-greedy mechanism,
formal task success, or independent generalization. Any later confirmatory
comparison needs a separate frozen method, source/disjointness proof,
resource contract, and prespecified statistics.
