# Shared selected-tail warm-start replan: post-hoc result

All eight roots and all five prefixes completed three replanning starts.
Every prefix on a root received that root's identical planner-selected tail,
then 480 CEM samples per start; the supplied tail remained an incumbent.
Alternate same-environment geometry audited the saved actions and scores,
and a separate audit verified all 120 returned incumbents were actually
scored.

The selected prefix's mean replan score was lower than both zero and greedy
on 6/8 roots; several differences were only a few thousandths and are not
evidence of a practically meaningful advantage. On root 0, selected/zero/
greedy means were 0.019299/0.502449/0.501454. On root 4 they were
0.012865/0.503993/0.501977, despite the original selected chain not meeting
the success threshold. These are post-hoc optimization diagnostics, not
independent task-performance estimates.

The result shows that exposing the same method-produced tail to every
prefix can change the replanning comparison. It does not identify the
specific cause of the zero-initialized CEM's failure and does not replace
the earlier intervention's original 0/8 mechanism intersection. The two
protocols differ in initialization and their outcomes must not be merged
into a gate claim.

Artifacts: `le-wm-main/results/pc_v22_modecount_common_tail_replan_posthoc_20261104_codex/`.
