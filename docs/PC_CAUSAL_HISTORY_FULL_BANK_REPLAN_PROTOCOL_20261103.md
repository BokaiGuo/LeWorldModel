# RGB history, full action bank and real suffix replanning

This is a new development experiment on exposed obstacle roots 8343, 8345,
8349 and 8351, not independent confirmation. Every branch shares two initial
zero actions (the observation prelude). Use actual consecutive post-step
RGB frames at the existing 0.05-second action clock to estimate
position and causal velocity; never use simulator qpos or qvel for planning.
The prelude is included in all physical-step and horizon accounting.

At the second prelude successor compute the same frozen-model 2 × 10 task response
and FD check as the five-action bank protocol. Form nine bounded five-action
prefixes using the zero, signed first two response singular directions and
signed first two null directions at infinity-norm amplitude 0.5. Basis
vectors and computed rank are saved; rank failure is an experiment failure.

For each prefix, optimize a five-action suffix with the same AP-compatible
CEM (16 samples × 3 iterations plus two final comparisons, 50 total), same
root-specific seed and frozen affine RGB-derived score. Score complete
two-packet model programs from the observed history. Define model eligibility
and selection using the existing `choose` rule: a nonzero prefix must have
non-improving predicted initial task score and its best two-packet score must
strictly beat both zero and the immediate-model-greedy prefix, otherwise no
selected prefix. Lock this result before executing any candidate branch.

Execute *every* prefix from the same original reset, with the same prelude,
and require all three initial images to match the ones used for planning. Then
use each branch's own actual final two RGB frames, its own resulting model
context, the same 50-sample CEM settings, and the same root-specific seed to
replan and physically execute that branch's own five-action suffix. Save all
candidate batches, actions, images, scores, first-packet score, selected and
control indices and per-branch planning counts. Candidate-bank outcomes are
for development and must not feed back into selection.

Independently replay all complete 12-action programs in the position backend
and report position at prelude, after each prefix and after each optimized
suffix, as well as the zero/same-bank-greedy/selected comparisons if a
selection exists. Compare immediate nonprogress against the *post-prelude*
physical position; report physical progress and model eligibility separately.
Do not manufacture a selection if none qualifies.

Report observation/prelude cost, 10 JVPs and 20 FD samples per root, all
model-predicted samples and actual physical and replay steps. A common
resource allowance is not established by sharing downstream suffix settings:
structure cost and selection overhead remain extra. No fixed common-suffix
intervention, native success, strong-baseline equal-total-cost comparison,
independent-root confirmation, or P1/PC admission is claimed.
