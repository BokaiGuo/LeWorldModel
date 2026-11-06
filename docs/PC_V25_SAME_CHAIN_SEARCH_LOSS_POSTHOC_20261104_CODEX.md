# v25 same-chain panel: post-hoc search-stage attribution

This analysis replays the eight sealed prospective plans using their
recorded predictor outputs, verifies every regenerated action batch and
the selected chain, and attributes each call to its actual v25 search
stage. No new physical trajectories were generated; this is a
same-code consistency and cost attribution check.

| Root | Global proposal best | Best before refinement | Final distance | Incumbent source |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.556556 | 0.503515 | 0.503515 | depth-2 candidates |
| 1 | 0.583255 | 0.523425 | 0.523425 | depth-2 candidates |
| 2 | 0.603022 | 0.506315 | 0.506315 | depth-2 candidates |
| 3 | 0.516580 | 0.506196 | 0.506196 | depth-1 successor probe |
| 4 | 0.548885 | 0.505022 | 0.505022 | depth-1 successor probe |
| 5 | 0.653410 | 0.504871 | 0.504871 | depth-2 candidates |
| 6 | 0.634675 | 0.593000 | 0.511387 | full-chain refinement |
| 7 | 0.549889 | 0.502290 | 0.502290 | depth-2 candidates |

Thus search improved on the global bank on all eight exposed roots,
but none reached the success threshold. Refinement strictly improved
the pre-refinement incumbent on only root 6. The 0.593000 figure on
that root is the minimum among *all recorded full-horizon evaluations
before refinement*, not just the retained beam.

Relative to state-diversity retention, the response rule added a
first-layer candidate on 5/8 roots; the recorded rule added none at
the second layer in this panel. On seven roots the final prefix exactly
matches a first-layer candidate, and none matches a response-added
candidate. On root 6 the final prefix changed during refinement, so exact
matching is inappropriate; its winning refinement start came from a
retained final beam whose first-layer ancestor was not the added
candidate. This is an observed lineage on the response run, not a
counterfactual demonstration of what a no-response search would do.

Mean prediction-token costs per root included 195 for the global bank,
882 for the three layers' finite-difference probes, approximately 140
for successor reachability probes, and approximately 341 for full-chain
refinement. Candidate scoring, initial rescoring and final comparison
consume the remainder. Prediction tokens do not represent SVD or runtime
cost. These observations locate where the budget went; they do not
prove a bug, a causal pruning failure, or an optimal search result. The
independently preregistered equal-total-token ablation remains the relevant
limited evidence for the net successor-probe comparison.

Raw attribution:
`le-wm-main/results/pc_v25_e0_prospective_same_chain_dev_20261104_codex/search_loss_posthoc.json`.
