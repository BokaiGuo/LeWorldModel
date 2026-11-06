# PC2 same-chain mechanism intervention for the frozen response arm

Date: 2026-11-04

This prospective development intervention consumes the actually selected
complete chains from the sealed `response` arm of the equal-slot retention
panel. It never injects a privileged prefix and never changes the selected
chain.

For each of 16 roots, the selected first five actions are compared with:

- zero prefix;
- greedy first-layer candidate;
- lookahead candidate selected by the sealed successor-value record;
- a deterministic random prefix.

All controls use the same model, exact Wall dynamics, target, action domain,
remaining horizon, common-tail probe set, and three CEM suffix replans with
480 prediction samples per repeat. The selected prefix and every control are
also measured with the actual full-tail float32 response interface at scales
0.01 and 0.05.

The predeclared mechanism quantities are:

1. non-immediate first-prefix progress;
2. nonzero selected prefix and tail;
3. task-relevant response change against every control;
4. lower common-tail relative score and lower mean suffix-replan score;
5. stable local branches for the selected chain and all controls.

Smooth response and contact/branch changes are recorded separately. A branch
label change is not treated as a smooth Jacobian. This is a PC2 development
intervention, not PC3, RGB closed-loop, source-independence, or generalization
confirmation.
