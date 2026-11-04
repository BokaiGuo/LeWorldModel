# Actual-successor RGB decomposition for fixed failure banks

After the fixed-candidate oracle result, use all 624 saved initial-plan and
fixed-prefix-tail candidates from roots 1889, 1893 and 1895. Replay their
exact ten actions with the legacy renderer; require every rendered terminal
position to equal the sealed position-only backend. No candidate may be
selected, dropped or regenerated using physical results.

Use the same fixed checkpoint, RGB goal and initial task mask as in the
original planner. Encode each true terminal RGB on host CUDA, compute its
masked latent goal cost and compare, within each bank, (a) saved world-model
score vs true-RGB latent cost; (b) true-RGB latent cost vs physical goal
distance; and (c) saved model score vs physical distance. Report pairwise
reversals/ties, chosen distances, and argmin indices. Save all true images,
scores and source bindings. Verify frozen model hash is unchanged.

Actual future images and physical outcomes are post-execution oracle inputs,
not deployable selectors. Initial and fixed-prefix-tail banks are different
experiments, adaptive candidates are dependent, and roots were selected
after outcomes. This cannot establish P1, independent effects, complete
code correctness, or a unique cause of failure. No model training.
