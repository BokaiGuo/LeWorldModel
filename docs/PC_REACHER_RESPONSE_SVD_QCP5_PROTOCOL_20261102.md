# Response-SVD all-prefix real-RGB Q-CEM development control

The previous all-prefix cp5 CEM used ordinary planning-coordinate L2 for the
suffix, while the response-SVD candidate bank used a goal-conditioned
task-weighted Q metric. This fixed follow-up replaces only the tail optimizer
objective with that same frozen goal-only Q: all 13 prefixes on each of the
same two already-exposed roots get an independent CEM from their actual cp5
RGB and action history. It retains the same candidate count, seed rule,
execution horizon, warmup, parameter-zero physical control and fixed-prefix
plan as the prior experiment.

For each root, the Q metric must come from that root's authenticated
pre-existing goal-only Gram matrix, not physical outcomes or simulator state.
The model computes both predicted and goal coordinates; Q weights their
difference. CEM uses the existing algorithm: 16 samples, two iterations,
four elites, seed root+5000, mean sample in slot zero, final best sample.
Every arm and failure remains in the durable journal. No candidate is selected
after observing physical outcomes. The old L2 run remains unchanged.

These are two known development roots and the method was proposed after the
L2 and common-tail results were opened. Paired comparisons are exploratory,
not independent confirmation or a pre-registered selection claim. A changed
objective can change search cost or numerical operations; 32 candidate
rollouts per arm do not establish equal total computation. This panel does
not establish a strictly model-selected P1 witness, strong-baseline advantage
or P1 admission.
