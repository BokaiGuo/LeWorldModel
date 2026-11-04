# PointMaze AP-compatible observed-target development experiment

Fix exposed roots 1888 and 1889 and their previously saved frame-10 goals.
Use observation-only memory episodes 1800, 1801 and 1802, frames 0–20;
reject query episode overlap. Embed memory RGB using the same frozen visual
encoder and mean spatial features, with zero proprio supplied only to that
visual encoding. Retrieve using the existing observation-memory key
`[current, goal, goal-current]`, horizon `max(5, 10-elapsed)`, and use the
actual retrieved RGB at start+5 as the AP target. Estimate its target
proprio from RGB with zero target velocity. The comparison arm uses the
final goal RGB instead.

Both arms use the same PointMaze model, raw-action box [-1,1], zero-raw
prior, AP-compatible CEM kernel with 16 samples, three iterations, four
elites and 50 scored blocks per decision, and equal seeds paired by root
and decision. Execute five distinct actions, refresh from real RGB with
causal velocity from frames 4 and 5, then retrieve/replan and execute five
more. No state, physical outcome or memory actions enter the selector.
State is allowed for initialization and post-run endpoint evaluation only.
Capture all RGB/action bytes and retrieval identities; count memory encoding,
retrieval, planning encoding and prediction separately and record wall
times. Independently replay all full action sequences in the legacy RGB
and position evaluator; fail closed on disagreement.

This is a small, previously exposed development study. Memory episode IDs
exclude the two queries but training membership and general independence
are unknown. Different memory access and offline memory cost prevent
equal-total-resource claims. This is an AP-compatible adaptation, not the
official AP memory, checkpoint, full controller or published evaluation.
Do not infer P1, program composition or independent generalization.
