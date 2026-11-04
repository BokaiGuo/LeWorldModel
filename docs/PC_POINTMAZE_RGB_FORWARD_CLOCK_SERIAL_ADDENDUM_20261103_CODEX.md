# RGB clock audit: serial-context addendum

Before the v2 run: retain the original frozen protocol and all acceptance
conditions. The first simultaneous-context attempt did not reproduce sealed
frames and its journal remains intact; the cause is not established.

V2 runs the two arms serially, each with a freshly created simulator, for each
of the same 64 frozen action sequences. It additionally records qpos, qvel and
time immediately after each arm's render-boundary forward (or corresponding
boundary without an extra forward) and requires exact equality to the recorded
state immediately before it. Both arms must match at every integration tick;
all 1,152 legacy frames must match the sealed hashes, and boundary physical
states must match the independently recorded prior replay. Any mismatch stops
the audit; no optimizer, model, or planner is called. The renderer must be the
NVIDIA host renderer. Failure is not evidence that the hypothesized cause is
correct.
