# Atlas strong-plan injection diagnostic

This is a development-only search/score wiring test, not Atlas performance
and not an independent experiment. Use the previously exposed eight unicycle
goals `2*pi*i/64+0.031`, `i=0..7`, and the exact v2 smoke configuration.
Obtain a two-action candidate from the existing four-start, 32-evaluation
analytical gradient planner for each goal. Then append its first packet to the
v2 root bank and its second packet to any node with the matching full
successor context. Never substitute a simulator outcome for model scoring.

For every goal record the injected chain, prefix score, full direct and split
model replay scores, witness hashes, cost rank among returned first-prefix
branch winners, whether its own branch retained that exact chain, whether the
overall selector selected it, and whether a different chain ranked strictly
better. A branch may legitimately retain a better suffix rather than the
injected suffix; distinguish that case from rejection. Direct replay and
prefix-context construction are diagnostic costs, separate from compiler
counts. Injection is privileged access to another planner's solution and must
not enter formal Atlas or be reported as method performance.

Preserve and compare historical v2 un-injected output. The injected run adds
one candidate branch and may add the suffix at duplicate contexts, so counts
are not matched. The metric is the model's declared weighted terminal mean
squared error; physical position error is recorded only for interpretation.
These exact dynamics are not a visual world model, real feedback, or evidence
of PC2/PC3 admission.
