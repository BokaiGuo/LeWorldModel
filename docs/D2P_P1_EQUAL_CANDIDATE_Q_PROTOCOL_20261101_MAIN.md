# Equal candidate suffix optimization — prospective development ablation

Retain the fixed eligible root0 return target and goal-only Q artifact from
the Q-return protocol. Run structured, native, random, immediate_progress once
each, in that order. Root1 remains initially-successful/ineligible; no substitute.
All failures stay in the denominator. No additional sample stopping rule.

Hypothesis: selecting after only two suffix updates and then giving the winner
32 updates can suppress prefixes requiring more suffix optimization.
Redistribute 8×2+32=48 updates into 8×6; remove winner-only refinement.
No other candidate generation, seed, learning rate, Q, prefix protection,
feedback time, execution horizon, or physical evaluator change.

This preserves optimization update counts for structured/native/random and removes
one redundant three-packet winner replay. Immediate_progress now receives48
instead of16 updates; explicitly report this increase. Structured still pays
Jacobian costs; equal updates do NOT imply equal total cost. Report category-wise
ledger counts and goal-metric construction13 encoder forwards/192 input VJPs.
Do not pad calls or claim superiority to certified external planners.

Compare all four trajectories and prefix cp0→cp5 pusher, objectXY, angle, jointXY
and native7 quantities. Native7 regression alone cannot certify non-greedy.
Same known development task, not independent confirmation. No gate changes.
Host CUDA only for neural work, exclusive outputs and nonblocking GPU lock.
Preserve both preceding Q and Euclidean runs irrespective of new outcomes.
