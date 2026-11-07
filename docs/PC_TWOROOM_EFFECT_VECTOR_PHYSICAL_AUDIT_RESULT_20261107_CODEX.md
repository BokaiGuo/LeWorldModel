# TwoRoom effect-vector physical alignment: post-hoc audit

This audit reuses the previously sealed common-tail model scores and physical
trajectories from 40 exposed roots. For each prefix it compares the ordering
of four A actions by predicted final latent goal-cost improvement against
the ordering by actual final goal-distance improvement. Both are relative to
the same zero-action tail, which is a legal reference action, not an asserted
physical hold. This is an ordering comparison across different units, not an
absolute effect-magnitude comparison.

Within each root, exact duplicate prefix action arrays are counted once.
Ties in physical effects within 1e-6 distance units are excluded; predicted
ties receive half credit. The task score averages its comparable prefixes;
the final summary gives equal weight to the 40 roots.

- Mean root-level pairwise agreement: **0.631**, range **0.200–0.880**.
  36 roots exceed 0.5 and four are below 0.5. These are descriptive counts,
  not a preregistered statistical test.
- For the descriptor-selected prefix, 31/40 roots had at least one
  physically distinguishable pair; their mean agreement was **0.933**.
  Nine roots have no comparable pair and are not silently counted as
  correct. Selection used the same model scores, so this is selection-biased,
  not independent predictive validation. Across all 40 selected prefixes,
  the numbers of comparable action pairs are 0 in nine roots, 3 in seven,
  5 in eight and 6 in sixteen.
- Five roots contained at least one prefix that had already succeeded by
  step 10; all roots had a mix of success and failure among A chains.
  Neither fact is used to exclude cases from the main summary.

The physical metric is endpoint distance, not shortest-path reachability or
the task success predicate. The analysis uses queried A actions, is post hoc,
and does not establish held-out prediction, descriptor benefit, a causal
reconfiguration mechanism or closed-loop gain. No gate is changed.

Two unit tests cover pair ordering, ties, missing-comparison handling and
invalid input. The audit verifies the original plan, replay and source
fingerprints before analysis. It does not rerun the physical simulator or
model.

[Audit code](../le-wm-main/experiments/pc_tworoom_effect_vector_physical_audit_20261107_codex.py) ·
[root-level data](../le-wm-main/results/pc_tworoom_effect_vector_physical_audit_20261107_codex.json)
