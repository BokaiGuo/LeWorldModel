# Development protocol: spectral prefixes with real RGB tail replanning

This exploratory run uses all four exposed roots 1880–1883 from the sealed
A1 spectral experiment. For each root, take its already selected spectral,
random-direction, and axis-prefix actions without alteration, and add a
zero-prefix control. Execute each five-step prefix from the same reset,
capture the actual native RGB at step 5, and encode it using the same frozen
checkpoint and fixed raw-zero proprio used in A1. The goal is archive RGB
frame 10. Each branch then evaluates the same eight fixed five-step constant
action packets from the previously sealed A0 candidate bank. Choose the
minimum predicted terminal latent MSE with first-index tie breaking, execute
that packet, and record the actual step-10 RGB. No physical state or physical
outcome is provided to the planner. Evaluate position relative to archive
frame 10 only after all four decisions for that root have been made.

Check the three inherited prefixes against their source RGB hashes, the
checkpoint and model hashes against A1, and the fixed bank source hash.
Record the goal, step-5 and step-10 RGBs and pooled latents, all eight
candidate costs, selected actions, and position-only offline replay. Compare
the endpoints by root, not by treating candidates as independent samples.

All branches use one shared goal encoding per root and one step-5 encoding
and eight tail predictions per branch. The A1 spectral prefix additionally
cost two JVPs and an SVD; the three A1 prefix selectors each scored five
candidates, while zero was fixed. These are inherited compute costs and
must be reported separately; equal tail budgets do not establish equal total
budgets. Finite-difference probes in A1 are diagnostic-only. Timing is
reported for this run and cannot reconstruct exact historical prefix costs.

The roots are exposed, the camera was selected using root 1880, the model
uses raw-zero proprio and the action normalization is from the entire
archive with unknown training provenance. Physical labels are from the
separate position evaluator, not the RGB bridge's hidden state. This is a
feedback development experiment, not a success-rate benchmark, complete
cost certification, reconfiguration mechanism demonstration, or P1.
