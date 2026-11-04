# Selected-chain random-bridge and lookahead intervention

Post-outcome diagnostic on all 24 already exposed roots. Bind the original
successor plans and intervention, the later control plans and audit, and
their source hashes. For each root take the actual final successor first five
actions and final ten-action suffix; also take the actual first five actions
of the previously sealed random-bridge and lookahead full methods after
refinement and fallback. These are full-method selected controls, not
necessarily the unrefined random bridge and lookahead candidates.

Apply to all three prefixes the same six ten-step suffixes: zero,
`(±0.5,0)`, `(0.5,±0.5)`, and the successor's actual selected suffix.
Report absolute terminal distances and changes relative to each prefix's zero
suffix. Then independently optimize each prefix's suffix using 120 full-chain
CEM samples, a common seed `2026121400 + root`, same optimizer and action
bounds. Recompute all saved scores, chain binding, and minimum-search scores.
Report all 24 roots and aggregate counts. The earlier selected-vs-zero and
same-bank-greedy intervention remains unaltered.

This diagnostic cannot turn the already negative original three-condition
screen into a PC2 pass, nor does it establish causal response reconfiguration:
these controls are post-outcome development choices and exact-state wall
dynamics are not the RGB model. No new planning, budget tuning, outcome-based
root filtering, or historical gate mutation is allowed.
