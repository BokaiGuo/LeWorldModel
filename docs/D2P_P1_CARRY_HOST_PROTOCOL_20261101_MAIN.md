# P1 real-feedback carry exploration

Both existing known roots 0 and 1 are development inputs, not independent
confirmation. Preserve every arm and every stopped or failed node.

Use the existing frozen DLS/Gaussian mixed candidate generator and optimizer.
At boundaries 0 and 5 retain the original H3 decision. At boundaries 10 and 15
rescore all eight optimized fresh plans and the preceding selected plan's
unexecuted tail from the same actual RGB context. Compare equal H2 then H1
horizons, using terminal latent MSE. Exact ties retain the old plan.

Apply this policy to all four arms. Additional per-arm costs at boundary 10:
18 action-encoder calls, 18 predictor calls/samples, 9 replay and 9 selection
calls. Boundary 15 costs: 9 of each of those axes. No additional backward.
All candidate generation and original optimization costs remain charged.
This is a larger exploratory contract, not an equal-FLOP comparison.

The actual executed packet must exactly match the preceding selected plan.
The normalized action history must match actual commands; current RGB is
provided by the existing metered physical episode. Seal the full plan before
dispatch. Retain early termination without padding trajectories.

Do not use physical outcomes to select candidates. This experiment does not
establish non-greedy progress, capability change, independence, or P1 admission.
Mechanism interventions and separately planned suffix controls remain required.
