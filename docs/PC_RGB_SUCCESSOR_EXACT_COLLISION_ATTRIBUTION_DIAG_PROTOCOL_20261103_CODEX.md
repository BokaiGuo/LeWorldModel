# Successor exact-wall collision and response attribution diagnostic

This post-outcome engineering diagnostic consumes only the already sealed
24-root policy plans, the prior response-scale audit, and exact wall dynamics.
It does not run or change a planner, select a task, change a finite-difference
scale, or reopen the stopped block-program protocol.

For each final selected chain, replay the precise float32 model actions for
15 steps and report collision steps, final distance and distance to the wall.
For each of the 72 first and successor response contexts, reconstruct the
actual nominal chain and all positive/negative coordinate perturbations at
0.01, 0.05 and 0.1. Record each chain's complete collision-event pattern.
For each context and scale, identify whether any paired perturbation has a
different collision pattern. Compare these event transitions to the
previously recorded response-matrix change from 0.05, reporting the full
2x2 counts for relative change above 0.1 as a descriptive association.
Check the original 0.05 matrices and selected scores against their sealed
records before reporting.

Analyze all roots and contexts. This diagnostic can establish whether wall
events coincide with a response-scale difference in this finite set, not
whether the event caused planning failure or whether wall behavior is an
implementation error. Zero-reference norms and numerical precision must be
reported explicitly. Results remain exposed-root development evidence and
are not PC0/PC1/PC2/PC3 certification.
