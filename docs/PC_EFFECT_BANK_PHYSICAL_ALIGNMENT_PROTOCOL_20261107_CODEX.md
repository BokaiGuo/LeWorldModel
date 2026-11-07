# Executable effect bank: fixed A probes and physical alignment

Use the previously exposed TwoRoom roots 10–49, each with eight sealed
prefixes and eight common A tails. Score all 64 probe chains plus eight
equal-duration zero-control reference chains per root using the frozen
RGB-only model before any physical outcomes are read. The evaluation end
replays exactly those 72 chains from the recorded task state.

Record per-prefix zero-control displacement, first-ten-step success and every
chain's final physical distance and success. Compare model predicted
reference-minus-probe score against physical reference-minus-probe distance,
excluding exact physical ties (1e-6 distance) and exact model ties (1e-6
latent cost). Report both micro counts and equal-root mean accuracy, plus
the stratum whose prefixes did not succeed during the first ten steps. The
scales differ, so no raw MAE compares latent cost with physical distance.
Zero-control static behavior is checked at the reference prefix endpoints;
it is not assumed for other states or environments.

Development exit for this specific component requires at least 65% equal-root
sign agreement on all eligible roots and 65% on the non-success-prefix
stratum, with at least 30 roots in each. This is a diagnostic screening rule,
not the PC1 gate; passing would only justify a separate fair search test.
All roots and the A-bank were exposed before this protocol, so results cannot
establish independent generalization or PC0–PC5 passage.
