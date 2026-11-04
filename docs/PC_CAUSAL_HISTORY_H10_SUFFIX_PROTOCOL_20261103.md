# Causal RGB history with ten-step suffix: exposed-root development

Use exactly the four exposed roots, two-step shared zero prelude, and the nine
five-action prefix programs already sealed in
`pc_causal_history_full_bank_replan_20261103_DEV01`. Do not use its physical
outcomes for selection. Reconstruct the same cp2 RGB context from saved
post-step frames and the same causal RGB velocity. The frozen checkpoint,
affine task score, action bounds and goal remain unchanged.

For each prefix, optimize a ten-action suffix (two five-action packets) with
the same CEM population 16, three iterations and four elites; compare the
prior and final mean, giving 50 full three-packet model samples per prefix.
Use the historical AP-compatible unprojected-elite update and bounded scored
actions. Use common root-specific random seeds across prefixes. Test the
generalized 20-dimensional kernel against the legacy 10-dimensional kernel
before execution. Apply the existing nonprogress selection rule to the
predicted first packet and terminal-at-15-actions score, and freeze the
choice before any new physical outcome. No qualifying candidate means none,
not a retroactive alternative.

Then replay each of all nine prefixes from the original reset, verifying
all saved observations through cp7. From each real cp6/cp7 RGB pair
replan its own ten-action suffix using the identical CEM, budget and seed;
execute all ten actions to cp17. This is one cp7 refresh, not two feedback
opportunities. Persist full actions, images, CEM candidate packets and
scores, optimizer selection and resource counts. Replay the same actions
independently with the position backend at cp2, cp7 and cp17.

Compare zero, same-bank greedy, any preselected candidate, and an explicitly
labelled outcome-exposed best physically nonprogress prefix. Physical
nonprogress compares cp7 position distance with cp2. These arms share
prefix bank and suffix budgets, but the D2P geometry/selection cost is not
matched to an independent baseline. Old five-step and new ten-step suffixes
have different task horizons and cannot be treated as an equal-time ablation
or proof that extra computation caused an outcome. No common-suffix
intervention, native task success, independent root, strong-baseline
comparison or gate admission is implied.
