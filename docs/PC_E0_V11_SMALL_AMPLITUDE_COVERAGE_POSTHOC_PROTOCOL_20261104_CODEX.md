# Fixed post-hoc coverage diagnostic for the v11 amplitude panel

After the panel has stopped, classify every already-scored 15-step float32
chain by whether its independently replayed trajectory ever crosses to `x>0`,
ends at `x>0`, collides with the wall, and reaches the prespecified `<0.05`
terminal-distance success criterion. Report totals and per-root counts for
each arm, the minimum distance among crossing chains when present, and whether
the selected chain crossed. Also classify five- and ten-step scored trajectories
separately; their counts are search exposure, not independent samples or final
outcomes. Do not generate or optimize any new chains.

Require the source audit to pass and bind its result hash; re-evaluate every
saved trajectory with the independent line-segment implementation, validating
its stored terminal position and score. Preserve all roots and failed runs.
This is a descriptive analysis of an already exposed developer panel, not a
new prospectively registered planning experiment, a causal attribution, or an
independent replication.
