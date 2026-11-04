# Native PushT contact witness: bounded development protocol

The preceding constructed target did not move the object. This separate
evaluation-only diagnostic tests one fixed, legal feedback policy on one new
seed, to determine whether a nontrivial 15-action target can be constructed.
It does not search or repair the stopped random-goal root.

Use the native runtime and the same fixed ten-action warmup as the prior JEPA
feedback experiment. Reset once with seed `2026110432`. For each of exactly
15 subsequent actions, read the current physical agent and object centre
positions and issue `float32(clip((object_xy-agent_xy)/100, -1, 1))`. This
privileged feedback law is for witness construction only, never a planner
input or method candidate. Save each commanded action, RGB, proprioception,
physical state, warmup, runtime and dependency hashes. Termination or any
exception is a failure, not a reason to choose another seed.

Reinitialize the same environment and replay the fixed saved 15 actions after
the same warmup. Require exact RGB, proprioception and state at all 26
boundaries. Predefine a nontrivial target as object-centre displacement
strictly greater than 20 pixels or absolute wrapped angle change strictly
greater than π/9 from the post-warmup state. This ensures the post-warmup
state does not meet the current success rule for the constructed target.
Report the physical displacement and angle regardless of whether this criterion
holds. The final witness state is itself reachable, but its images and actions
must be separated: only the final RGB may later be supplied as a goal; physical
state and witness actions are evaluator-only. Stop after this one seed and
policy, whether or not a nontrivial witness is found.

This is a deliberately constructed development goal, not evidence that the
original random goal is feasible, a planning score, contact mechanism, or any
gate. Replay on the same native backend is consistency verification, not
independent scientific replication.
