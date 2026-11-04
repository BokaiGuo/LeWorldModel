# Exact successor policy: additional development controls

This protocol is fixed after the existing successor results were observed and
before the control runs. It uses the same 24 exposed wall development roots,
480 predicted action-packet tokens per method, exact wall predictor, fixed
15-step terminal distance, and the selected successor records from
`pc_rgb_successor_exact_same_policy_dev_20261103_codex_plans.json`. This is
not independent confirmation, a pre-outcome comparison, or a strong-baseline
certificate. The stopped six-dimensional block-program protocol is not
reopened.

Run the following four additional arms:

* `same_bank_greedy`: regenerate the actual first bank with the identical
  one-nominal response generator and charge its 21 five-step predictions and
  SVD. Evaluate both first prefixes at five steps, choose the lowest-distance
  prefix and optimize its ten-step suffix.
* `random_bridge`: regenerate and charge the same first bank. For each first
  prefix create zero and one fixed-seed random second segment, for four
  ten-step prefixes; optimize all four suffixes.
* `lookahead`: use the exact same generated random-bridge bank, score its four
  ten-step prefixes, retain the lowest-distance one, then optimize its suffix.
* `full_cem`: optimize the complete 15-step action sequence directly without
  generating a first bank; use the entire affordable budget.

For structured arms, reserve three tokens for final comparison, use three
quarters of remaining affordable full-chain samples equally over retained
prefixes (remainder goes to earlier entries), and spend the rest on the same
joint full-action CEM refinement with incumbent retained. Compare the final
chain to its zero-first-five-step counterfactual using one full-chain
prediction, and take the counterfactual on a tie. Full CEM reserves and
spends the same three tokens on a rescore of its selected chain, with no
abstention claim. All methods count generation and selection predictions;
SVDs and wall time remain separate. Random actions are iid uniform [-1,1]
with seed 2026121200 + root; optimization seed is 2026121300 + root and
refinement seed one larger.

Seal every prediction batch and verify action casts, exact scores, selection,
resource usage, actual first-bank agreement with the successor plan, and
dependency hashes. Report all 24 results and pairwise wins/ties/losses. Count
distance below 0.05 as success. Do not add seeds, tune root-specific
parameters, change the successor outputs, or claim equal total compute.
These controls do not themselves complete selected-chain random-bridge or
lookahead mechanism interventions, which are a separate requirement.
