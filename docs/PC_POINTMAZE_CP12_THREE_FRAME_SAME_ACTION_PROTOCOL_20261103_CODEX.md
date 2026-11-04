# CP12 one versus three observed frames: fixed-action development protocol

Frozen before model execution. Use the existing 32 executed branches in
`pc_causal_mixed_rejected_physical_20261103_DEV01`, four exposed roots and
eight branches each. No new candidate search, physical steps or outcome-led
selection. Retain duplicates and report root-level as well as branch-level
values.

At decision time 12 the one-frame arm receives RGB[12]; the three-frame arm
receives RGB[2,7,12]. Each frame's proprio input is the same causal RGB
position and preceding-one-command difference velocity estimator used by
the cp12 refresh, computed from RGB[t-1:t+1], dt=0.05 s. Both receive the
identical executed outgoing raw actions[12:17], normalized once and packed
chronologically. The three-frame action tokens are actions[2:7], [7:12],
[12:17] respectively. The predictor takes its final output as t=17; both
arms use one forward predictor sample per branch. Past observation encoding
cost is additional for three frames and must not be called equal compute.
Simulator state and future RGB are never model context or selection input.

Use saved RGB[17] only after prediction as an evaluation target, encoded
with causal RGB proprio at 16/17. Primary diagnostic: full visual latent MSE
against this target; secondary: proprio latent MSE, paired differences and
agreement of the one-frame prediction with the existing saved cp12 refresh
arrays. Save per-branch predicted and target tensors, input states, action
packets, clocks, model parameter hashes and source/dependency hashes.
Report mean and counts of lower errors, per root; no significance or
independent generalization claims from these dependent, selected branches.
The checkpoint's original training data/configuration provenance is not
certified. This is conditional offline prediction, not a planner, physical
improvement, an unbiased action bank, or independent confirmation. Causal
RGB velocity is interval averaged and historical render time may be offset.

Fail closed on output existence, source hashes and roster, non-finite data,
invalid action bounds, packet/shape/clock mismatch, checkpoint mismatch and
failed one-frame saved-prediction parity. Do not modify existing artifacts.
