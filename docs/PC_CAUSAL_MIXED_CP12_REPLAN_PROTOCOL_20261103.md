# cp12 same-prefix real-RGB replanning development

Use all 32 saved exposed-root mixed candidate trajectories in
`pc_causal_mixed_rejected_physical_20261103_DEV01`. Re-execute the exact
first 12 physical actions and require all 13 RGB frames through cp12 to match
the saved source. Reconstruct causal state from real cp11/cp12 RGB, encode
cp12, and optimize the last five actions with `packet_cem` (16 samples,
three iterations, four elites; 50 scored candidates), initialized at that
trajectory's original last-five-action tail. Bound actions in the same
normalized and physical domains. For each root use common RNG seed
`2026110410 + root` across prefixes. Choose between original tail and final
CEM mean by the frozen task score; execute the selected tail, even on a tie.
Persist CEM proposals/scores, final actions, all RGB frames, task scores and
the chosen/unchanged flag.

Re-evaluate cp17 physical distance using the same separate position backend.
Compare every new branch with its source trajectory and root's zero/greedy
references. Encode the final real RGB-derived state for a same-action
prediction-versus-observation diagnostic. Track already saved cp12 physical
context, and fail closed on RGB mismatch or invalid binding.

This adds a real observation and an extra optimization at cp12; the source
policy did not have that opportunity. It tests a feedback intervention,
not equal-budget superiority over source or baselines. Any best candidate is
outcome-exposed, not a previously selected policy. No common-suffix mechanism,
new independent root, native success or PC gate is claimed. Raw physical
state is evaluation-only, and actual action history is not explicitly input
to the one-frame encoded context. Record all added and upstream costs.
