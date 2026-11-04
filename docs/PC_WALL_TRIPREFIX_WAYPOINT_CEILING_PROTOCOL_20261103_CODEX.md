# Triprefix 12-step waypoint ceiling diagnostic

After viewing the triprefix CEM results, use the identical three saved
prefixes, three seeds and exact wall backend. This is an exposed-task
diagnostic, not a Planning Compiler arm, RGB-only method, independent
confirmation or gate. Its generator uses exact prefix-end pose and known
unicycle kinematics; neither may be treated as deployable RGB information.

For each prefix generate 256 waypoint suffixes of 12 actions by drawing the
same seed-indexed 256 points from the previously used task-scale box
`[-0.5,0.5] x [-1.5,1.5]`. Construct each bounded program from the exact
prefix endpoint to the sampled point and then the goal: turn in-place using
the minimum bounded turn commands, advance in straight lines using the minimum
bounded 0.3 translation commands, truncate after 12 commands or pad with
zero. Construction ignores wall geometry and collision outcomes. Execute
each proposal with exact collision dynamics and select minimum terminal
distance, breaking ties by earliest index. Replay the full 16-action chain
and require the distance to match the suffix rollout.

All three arms get the same samples, suffix transition count and horizon as
the preceding CEM arm, but use a privileged different search parameterization
and different overhead. Count prefix/suffix/replay separately; no claim of
equal end-to-end compute or method superiority follows. Report all prefix
states, final errors, counts of candidates below 1e-8, selected waypoints,
and cost for all seeds, including failures. Bind input/source hashes and use
a new exclusive receipt. Compare this ceiling with prior CEM only
descriptively and retain all old outcomes.
