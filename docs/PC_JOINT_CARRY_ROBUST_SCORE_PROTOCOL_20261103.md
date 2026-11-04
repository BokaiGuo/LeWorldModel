# Fixed-bank action-perturbation scoring: exposed development

On all 17 already exposed source roles, compare the saved original joint
tail against the saved carry-CEM tail from the identical real cp7 RGB
context. This is an offline fixed-bank diagnostic, not a new closed-loop
trial. Source outcomes have already been examined; no confirmation or
independence claim is possible.

For each of the two ten-command raw tails, evaluate the same fixed five
model inputs: the original tail and adding `+0.05` or `-0.05` to the first
or second action coordinate at *every* tail time step. Clip each raw
perturbed action to `[-1,1]`, normalize once with the same source action
statistics and use the same frozen two-packet model and goal-projected
proprio score. All five perturbations are equally weighted. The perturbations
are robustness probes, not physically executed trajectories.

Freeze two predeclared decision rules: (1) nominal lower score, tie old;
(2) lower mean of five scores, tie old. Score all 17 pairs in source order;
save and hash the complete 34×5 score array, raw perturbed actions,
source/dependency hashes, model-before/after hash and actual call counts
before executing the separate outcome-evaluation stage. Verify nominal
old/carry scores reproduce the source's selected scores to `rtol=3e-5,
atol=1e-6`; fail on context, action, normalizer or nonfinite mismatches.
Only after the score receipt is written may the evaluation stage compare
both fixed decisions to the already saved position-replay distances. Report
all wins/ties/losses and per-root decisions, including duplicates, not just
the four adverse examples. No physical state enters either scoring rule.

Charge the robust rule for all ten two-packet prediction samples per pair;
nominal scores are part of that batch, so this does not establish
equal-total-compute superiority. The tested perturbation pattern and
amplitude are fixed development choices, not a calibrated uncertainty
estimate or an invitation to tune after results. Real cp7 contexts were
obtained by executing distinct prefixes; comparing across roles is not a
deployable prefix selector. The results cannot establish independent
generalization, P1 mechanism, or improve historical outcomes retroactively.
