# Goal-Q selected prefix: real-feedback development follow-up

Use the entire sealed model-only output
`reacher_pc_goal_q_selection_20261102_v1/report.json`, bound to its root source.
On each of both known roots execute Q-selected, zero, Q-immediate-greedy, and
both original appended refined candidates. If the Q selector returned no
eligible candidate, its physical arm uses zero and must be reported as an
ineligible zero fallback, not as a selected reconfiguration. Do not remove
duplicate arms or choose a different candidate based on physical outcomes.

Use the existing five-step prefix / real RGB and actual action-history
refresh / same two-token 16-sample, two-iteration CEM suffix execution for all
arms. This suffix CEM optimizes the original Euclidean model cost, not Q.
Record exact actions, per-arm stop status, qpos metrics, model identity and
separate compilation/selection/suffix cost. This is an exploratory known-root
comparison, not an equal-total-budget test, common-suffix intervention,
independent confirmation, or gate update.
