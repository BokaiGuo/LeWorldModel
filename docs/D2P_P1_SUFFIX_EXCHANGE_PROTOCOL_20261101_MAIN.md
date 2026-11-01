# Prospective suffix exchange, fixed four-arm comparison

Use the same eligible root0 return task, frozen checkpoint, goal-only Q, seeds,
six optimization updates per candidate, fixed initial prefix, feedback boundaries
and native evaluator as horizon-consistent Q. All four existing arms are run once,
including unsuccessful arms; no new roots selected using outcomes.

At H3/H2 nodes, form all64 combinations of the eight optimized first packets and
eight optimized tails. Evaluate each nonlinear rollout with the same model/Q.
Keep the best tail for EACH prefix, retaining its own on exact ties, then choose
the best complete plan by model score. Preserve the original eight combinations.
At H1 there is no tail to exchange, so do not make64 redundant calls.

Every arm receives the same exchange opportunity. Add64×H forward/action-encoder
calls and64 replay/selection events per active node:320 extra predicted packets
over a full20-step trajectory, no extra backward calls. Account for these before
dispatch and validate against hooks. Jacobian costs still differ by arm.
This is NOT a fair-total-cost improvement over the old lower-budget version.

No real successor labels enter selection. The method is motivated by earlier
developmental diagnostics, so the task remains developmental, not independent.
Check whether the chosen prefix changes, whether immediate physical progress
remains, native success/stopping, and actual source/cost integrity.
Do not force candidate3, alter gate, train weights or promote a common-tail
effect into a planning-capacity claim. Neural experiments on host CUDA only.
