# Common-suffix intervention for all feasible development prefixes

Use both roots and all five fixed prefix/tail action witnesses from
`reacher_pc_feasible_prefix_physical_20261102_v1/report.json`. For every
root execute the Cartesian product of all five prefixes with all five saved
ten-step tails and a ten-step zero-action tail: 30 branches per root.
The family includes previously outcome-exposed actions and is not selected
or reduced based on any result. Record stepwise qpos, qvel, target, reward,
termination and truncation; stop and retain actual length on censoring.

Verify the source report hashes, root/goal images, action bounds and each
replayed prefix's qpos distance against its own saved execution. For each
prefix and tail report final absolute qpos distance and its change relative
to the same prefix's zero-action tail; do not invent values after termination.
Check whether own-tail intervention branches match the already sealed
physical run. These comparisons measure an action's effect, not the
optimized controllability of a state or an independent mechanism claim.
Do not select a post-hoc winning prefix for P1 or use these roots as fresh.
