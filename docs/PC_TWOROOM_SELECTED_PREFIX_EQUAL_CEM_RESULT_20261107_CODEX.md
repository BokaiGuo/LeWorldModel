# Selected-prefix equal-budget tail CEM: development result

On all 40 exposed roots, three fixed prefixes received the same 4 × 64
complete-chain model evaluations, same four initial B tails, zero tail,
random-number stream, optimization schedule and selection rule. GPU plans
were sealed before physical truth was opened. All 120 selected chains were
then physically replayed from the matched reset; initial RGB matched.

| Fixed prefix | Optimized selected chain successful | Selected tail from original four B tails |
| --- | ---: | ---: |
| Original descriptor-selected | 39/40 | 34/40 |
| Zero | 0/40 | 0/40 |
| Original actions in reverse order | 34/40 | 27/40 |

Original versus zero has 39 exclusive wins and zero losses; original versus
reversed has five exclusive wins and zero losses. These are descriptive
paired counts, not independent confirmation.

The prior fixed-tail intervention succeeded on **40/40** roots for both
original and reversed prefixes. Therefore the reoptimizer selected a
physically failing tail despite at least one available successful B tail on
one original-prefix root and six reversed-prefix roots. The original actual
selected chain is not the same object as the post-hoc reoptimized chain.
This demonstrates that model-score-based reoptimization can lose an available
physical solution in this development setting; it limits any interpretation
of the 39 versus 34 comparison as evidence for an order-dependent mechanism.
Immediate prefix distance progress is also still a confound.

All arms made 256 complete-chain queries per root, **30,720** across the
experiment. Synchronized scoring calls took approximately 10.49 seconds in
aggregate, excluding model loading, generation outside score calls, the
original prefix discovery and physical execution. GPU cost is not a complete
system comparison. Two targeted tests passed for fixed prefix, action bounds,
initial-tail inclusion, query count, best-sample selection and invalid input.
Gym emitted the same observation-type and action-casting warnings.

This is an optimizer- and budget-specific comparison on exposed tasks. It
does not establish non-immediate reconfiguration, descriptor discovery,
independent performance or RGB feedback benefit. No gate was changed.

[Protocol](PC_TWOROOM_SELECTED_PREFIX_EQUAL_CEM_PROTOCOL_20261107_CODEX.md) ·
[code](../le-wm-main/experiments/pc_tworoom_selected_prefix_equal_cem_20261107_codex.py) ·
[sealed model plans](../le-wm-main/results/pc_tworoom_selected_prefix_equal_cem_plan_20261107_codex.json) ·
[physical traces](../le-wm-main/results/pc_tworoom_selected_prefix_equal_cem_result_20261107_codex.json)
