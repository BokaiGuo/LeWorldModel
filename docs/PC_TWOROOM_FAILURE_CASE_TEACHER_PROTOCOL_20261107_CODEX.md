# Failure case packet and observation alignment

Retrospective diagnostic on the same already-selected roots 38 and 39 and
their 65 sealed chains per root. Use the existing stored predicted and
observed five-step representations, actions, RGB frames, and physical
replays. Confirm scaler inverse round-trip, chronological five-by-two action
packing, initial reset-render versus planning RGB and encoder mismatch.
Check one-packet predictor with planning initial embedding and first packet
reproduces the first stored autoregressive prediction.

Then, evaluator only, form a single-next-packet prediction at every packet
boundary using the last at most three real encoded observations and the
corresponding recorded normalized action packets. The first observation is
the original planner input, subsequent observations are true rendered
successors. Compare per-transition squared-feature MSE against stored
autoregressive prediction and the same true next observation. Save all
predictions and measures with content-hash bindings. Report selected chain,
model-best B and best successful B plus all 65-chain mean and medians, by root
and packet; these are correlated exposed development cases, not independent
tasks or a closed-loop performance trial.

Teacher forcing consumes future actual observations and is not a deployable
open-loop planner. Relative improvement cannot identify the unique cause,
establish an online benefit, certify hidden-state equality, or change a gate.
