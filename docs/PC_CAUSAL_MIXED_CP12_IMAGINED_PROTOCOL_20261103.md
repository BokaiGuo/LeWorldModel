# Matched cp12 feedback versus imagined context

On the same 32 candidate and eight zero/greedy exposed-root trajectories
used by the cp12 real-RGB replanning experiments, re-execute and verify the
same first 12 physical actions and RGB frames. Encode the actual cp7 RGB
and causal estimated state; roll the frozen model forward with the saved
cp7–cp12 five physical actions to form an imagined cp12 context. No cp12
true RGB or physical state is used by this planner, though the physical
replay is verified for binding.

At that imagined context optimize the original five-action tail using the
same `packet_cem` population, iteration count, original-tail initialization,
root-specific seed and score as the real-RGB cp12 intervention. Execute its
selected tail from the actual physical cp12 state. Save candidate scores,
actions, images and physical endpoint. Compare same-branch outcomes with
the previously saved actual-RGB cp12 replan and original old tail.

The two cp12 planners have the same 50 optimization samples and identical
physical prefix and action budget. The imagined variant additionally uses
one five-action model prediction; the actual variant consumes a real cp12
RGB observation. Log both costs. This isolates the context intervention at
the prescribed planning interface more cleanly than comparing with no
optimization, but does not match total compute, prove independent advantage,
or remove RGB state-estimation and model-score limitations. The observed
states are not independent; selected-prefix performance, common-suffix
mechanism and PC gates remain separate questions.
