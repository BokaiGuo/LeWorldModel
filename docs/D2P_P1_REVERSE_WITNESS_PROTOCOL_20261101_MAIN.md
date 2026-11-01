# Prospective opposite-side push task witness

The current return task's object displacement points away from the initial
pusher, consistent with direct pushing. This motivates a separately labeled task;
it does not invalidate old outcomes or prove that all direct pushing is optimal.

Use only known root0, exact original reset/warmup replay. Before observing new
outcomes, fix a20-action waypoint witness: relative to initial object center,
(+130,-100) for4 steps;(-100,-110) for6;(-70,-70) for3;
(+45,+45) for4;(+130,+80) for3. At each step use
clip((waypoint-current_agent_position)/100,-1,1). This privileged scripted
controller is solely a task generator/reachability witness, not a baseline or
input to the learned planner. Record every action, state, native stop flag and RGB.
Honor native stop/domain guard; no reset-state editing or hidden continuation.

The observed final state/RGB is the proposed target. Admit this task for later
planning experiments only if it has a complete20-action public witness, is not
initially successful under the unchanged native predicate, moves the object
at least20 pixels toward the initial pusher (positive displacement dot product),
and its witness cp5 leaves object position/angle unchanged within1e-8 while
jointXY error to the final target does not decrease.

These are task-design eligibility checks, NOT P1 method evidence. The world-model
planner must independently choose its actions using RGB and its existing inputs;
it must not consume the scripted route or true state. No new-root independence
claim. If generation fails a check, retain the result; do not tune this route
under the same protocol. No neural inference in task generation.
