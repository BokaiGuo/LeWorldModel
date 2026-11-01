# Fixed normalized-gradient exchange comparison

Run all four arms on the same eligible return root, after suffix-exchange protocol.
Only change suffix update from alpha*g to alpha*g/max(abs(g)), alpha=.05 retained.
Clamp raw actions to[-1,1], retain zero gradients, and retain existing initial
prefix gradient mask and unused-tail zero gradient check. Six updates, no extra
rollout/gradient calls, same64-combination exchange, seeds and actual RGB schedule.

Motivation: Q loss scaling can make raw-gradient updates small relative to the
bounded action range, preserving initialization. This is an optimizer hypothesis,
not an already-proven code error. Log gradient infinity norm, actual peak action
change and pre-update model loss for all48 updates per node. Normalized steps may
overshoot; no descent guarantee, hidden backtracking or tuning grid.

Same model-call budgets as suffix exchange for each corresponding arm; structure
costs still differ across arms. Do not claim identical total cost across arms.
Retain all four outcomes, exact stops and physics metrics. Unchanged success and
P1 gate. Known developmental task, no independent confirmation or training.
Host CUDA neural execution only, exclusive artifacts, no old output replacement.
