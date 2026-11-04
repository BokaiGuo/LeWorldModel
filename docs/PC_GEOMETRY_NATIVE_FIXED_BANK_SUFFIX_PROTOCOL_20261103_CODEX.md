# MuJoCo 2.1.2 fixed-bank suffix development diagnostic

This is a new protocol. It supersedes neither the unrun 3.1.6 suffix
protocol nor any prior outcome. Use the same four geometry-defined starts,
zero initial velocity, fixed goal `(1.8,2.8)`, official executor XML and
the persistent MuJoCo 2.1.2 action clock. The output is privileged
simulation-only diagnosis, not a deployed RGB planner or independent test.

The position-only, five-action prefix bank is fixed before observation:
zero; a privileged one-step direct-goal greedy controller constructed
afresh on the 2.1.2 backend with eight directions and four magnitudes; eight
constant unit directions; and two independently sampled uniform `[-1,1]`
five-action random bridges. The same random bridges are used across roots.
Keep every candidate on every root.

For each root and prefix, apply the same six ten-action suffixes: zero and
five uniformly sampled sequences. Report initial and prefix-end distance,
all prefix states, final distance and terminal distance change relative to
that prefix's zero suffix. Declare physical non-improvement only when the
five-action endpoint distance is at least the initial distance.

Independently optimize a ten-action suffix for each prefix using privileged
position-only terminal distance and a fixed common-random-numbers CEM:
two starting means (zero and a random uniform sequence), three generations
of 16 samples, four elites, initial standard deviation 0.5, minimum
standard deviation 0.02, bounded actions, then score each final mean and
the zero suffix. Each prefix receives 100 full-trajectory evaluations. Use
identical random draws for prefixes within a root, with deterministic RNG
seed 2026110318. Record every physical evaluation and its action sequence.

All chains execute the full 15 actions without an early stop; endpoint
distance is the diagnostic metric, not native success or hitting time.
Replay from the original reset for every evaluation. No simulator-derived
result is made available to an RGB method. The 100-per-prefix diagnostic
does not represent an equal-total-cost comparison of candidate generation
methods or a proof of an optimal suffix. In particular, direct full-horizon
CEM, ordinary lookahead and random-bridge methods remain necessary under a
separately fixed total-resource policy before making a performance claim.
Fixed full-horizon baselines and closed-loop feedback remain outstanding.
