# Task 1 endpoint factor intervention

Use the previously sealed task 1 shared-prefix pool's original selected chain
and its sole final-success chain. Their physical step-25 object poses and
pusher positions define a 2-by-2 static factorial: each object pose crossed
with each pusher position. Reset a fresh environment with zero velocity for
every cell, set the original target, and render the result. Check observed
pusher and object pose against requested pose (absolute tolerance 1e-4);
any mismatch makes that cell invalid. The same-pair images' frozen encoder
embeddings must agree with previously saved real-endpoint encodings to 1e-5
absolute and relative tolerance or interpretation stops.

Score all four images against the unchanged original target-image embedding.
Report each score, the two pusher changes at fixed object pose, the two object
changes at fixed pusher position and the factorial interaction. Record images
and actual physical states. Counterfactual states can overlap, be out of
distribution, or be unreachable; rendering can contain occlusion and therefore
the results are not a clean latent-factor decomposition. Simulator state and
future images are strictly evaluation-only. One post-selected task cannot
establish a general method, RGB-only control benefit, or any gate.
