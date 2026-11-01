# Fixed push-and-return development tasks

Fix two tasks before evaluating any new planner actions: for each existing root,
take its historical cp10 object position and angle, but require the pusher's final
position to equal that root's initial position. Preserve initial pusher velocity
components in the native state7 target; the original native success predicate is
unchanged. Neither ignore pusher error nor change gate thresholds.

Generate targets with the original environment, not image edits. Native `_set_state`
advances physics, so resetting to state7 alone cannot reproduce a moving snapshot.
Full snapshot restoration plus shape reindexing still differed by three pixels
on root0 and was REJECTED; no tolerance was relaxed.

Use literal historical replay instead: original two-reset placement, exact ten
warmup commands and exact ten original witness commands. Verify all three history
images/states, each witness state/action/termination, and exact final cp10 pixels.
Then change only pusher position/velocity to the root values, reindex its shapes,
verify new state7 and render. Record resets, implicit setup steps, twenty public
steps and static target placement. These are replayed historical actions, not
new planner successes. Reachability of the new goal within20 remains unknown.

Encode exactly two target images with the existing frozen model on host CUDA.
Save raw pixels, image hashes, state7, model hashes and sources. Simulator state
defines the evaluation task only; the planner receives its target RGB embedding,
not object/pusher coordinates or physical outcome labels.

The intended challenge is temporary actuator displacement before returning, but
this is NOT an assertion that non-greedy mechanism exists, that the planner will
choose it, or that the task is feasible within the current horizon. Existing
diagonal and cross-goal failures remain part of the record. These two tasks are
development-only, not independent roots or a substitute for strong baselines.

Pre-control qualification found root1 initially successful (joint XY error
14.07462663); root0 is not (101.67639904). Retain both target records and their
eligibility flags. Root1 is rejected for control evaluation, not replaced and
not counted as a planning success. This is the original initial-success guard,
not a relaxed threshold. Record the two-task attempted design and one eligible task.
