# TwoRoom selected-prefix common-tail intervention

The predeclared development protocol replays all 40 exposed roots, the
descriptor arm's previously selected prefix, and four identical B tails,
with selected, zero, and reverse-order prefixes. It produced 480 physical
chains. The selected-prefix chains reproduced all prior full-matrix traces
within the declared distance tolerance; initial RGB matched on every reset.
Two tests passed for shared tails, controls, action bounds and finite inputs.

| Prefix | Selected B tail, roots successful | All four B tails, successful chains | Roots with any successful B tail |
| --- | ---: | ---: | ---: |
| Selected | 40/40 | 140/160 | 40/40 |
| Zero | 0/40 | 0/160 | 0/40 |
| Reverse action order | 40/40 | 140/160 | 40/40 |

For the selected tail, selected versus zero has 40 exclusive wins and no
exclusive losses; selected versus reversed has neither. None of the prefixes
had already succeeded by step 10. Mean prefix-end goal distance was 49.19
for selected, 59.91 for zero, and 50.38 for reversed. Distances are evaluation
metrics, never planner inputs. The 160 chains share 40 roots and are not 160
independent tasks.

The zero intervention demonstrates dependence on doing something during the
first ten steps, but the selected prefix makes immediate distance progress
and reversing its order does not change success. Thus it does not isolate a
non-immediate reconfiguration mechanism or show descriptor-driven discovery.
Equal-budget tail reoptimization, independent roots and actual online
selection remain absent. PC2 and other gates remain unchanged. Gym emitted
the existing observation-type and action-casting warnings; replay used the
same simulator as the original.

Additional audit of the existing candidate bank: 26/40 roots contain repeated
identical prefixes; their distinct-count histogram is 8:4, 9:2, 10:9, 11:6,
12:5, 13:14. The old lookahead retained two identical prefixes in one root;
neither other arm did. Identical complete chains had up to 0.000168 difference
in model scores across the original batch arrangement. This is a real
candidate-identity/slot-accounting limitation, not evidence that fixing it
would yield a descriptor advantage.

[Protocol](PC_TWOROOM_SELECTED_PREFIX_COMMON_TAIL_PROTOCOL_20261107_CODEX.md) ·
[runner](../le-wm-main/experiments/pc_tworoom_selected_prefix_intervention_20261107_codex.py) ·
[raw traces](../le-wm-main/results/pc_tworoom_selected_prefix_intervention_20261107_codex.json)
