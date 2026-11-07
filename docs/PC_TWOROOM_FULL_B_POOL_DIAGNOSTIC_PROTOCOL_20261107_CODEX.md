# TwoRoom fixed B pool diagnostic

Development-only, previously exposed roots 10–49. Freeze and score all 64
existing B actions per root using the RGB-only frozen model before any
physical replay. Score in batches of 32; compare previously queried B scores
and record numerical differences. Stable index breaks score ties.

Then replay every fixed action from its recorded reset and goal for all 25
environment steps, including after termination, using the previous evaluator.
Record each step's distance and termination. Report ever success, final
success, and success after a prefix of ten steps that had not already
succeeded. For each root compare full-pool existence, previously queried
coverage, the model-best B outcome, the best-ranked successful B and its
score relative to the previously scored best incumbent. Report actual
previously selected outcomes separately.

This is a retrospective evaluator-only oracle, not a planner, a new search,
or an independent confirmation. Physical information cannot affect scores,
selection, or future evaluation tasks. No gate is promoted. A successful
unselected action indicates an opportunity in this fixed pool, not evidence
that a practical method can select it. A pool without successes does not
establish that other generators cannot succeed. Model-score disagreement
alone does not distinguish dynamics from target-score errors.
