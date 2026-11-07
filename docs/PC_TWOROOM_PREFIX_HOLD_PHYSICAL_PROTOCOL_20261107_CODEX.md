# TwoRoom post-prefix hold qualification

Replay the existing four exposed seeds and five fixed five-step prefixes.
From each actually reached state execute five zero actions and record position
after the prefix and every zero-action step. Compare the full physical distance
trace to the previously sealed prefix/zero-tail record; report position drift
at each prefix separately. Treat maximum drift at most `1e-6` position units as
stationary in this local deterministic backend only. Do not infer that zero is
a hold at the initial state or at other states, environments, or checkpoints.
The known seed 2717103 reset-border exception stays in the denominator.
Physical state is evaluation-only and is not fed to the RGB planner.
