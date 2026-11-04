# Joint prefix/suffix gradient search: exposed-root development

Use the same four exposed roots, cp2 RGB context, frozen checkpoint, action
normalizer, affine RGB-estimated task score and five-action packet convention
as the causal mixed-prefix experiment. Reuse its sealed four raw and four
spectral initial prefixes, plus the same zero and source-greedy prefixes.

Optimize each 15-action plan against the nonlinear three-packet model:
12 Adam updates in raw action coordinates, learning rate 0.05, beta
coefficients 0.9/0.999, epsilon 1e-8, projected per action to [-1,1].
The batch objective is terminal cost plus
`10 * relu(initial_cost - predicted_prefix_cost)^2`. Prefixes of zero and
greedy are fixed; all other prefixes and all suffixes may change. Every plan
starts with two zero action suffix packets. Record the initial plan and all
12 updated plans, their first and terminal scores, and objective gradients.
The objective is a search aid, not a replacement for the strict constraint.

For zero and greedy use the best terminal model score among their 13
iterates. For each non-control candidate retain its lowest-terminal iterate
whose first-packet score is at least the current score. Separately for each
raw/spectral arm return the lowest-terminal feasible candidate only if its
predicted terminal score is strictly lower than both controls; otherwise
return `null`. Also retain each arm's unconditional lowest-terminal chain
among its four optimized candidates plus the two controls as an exploratory
performance winner. Decisions are fixed before physical outcomes.

From each root execute zero, greedy, raw unconditional winner, and spectral
unconditional winner, plus each non-null qualified candidate if not already
one of these four. Execute the chosen five-action prefix, use its real cp6
and cp7 RGB frames to reconstruct cp7 state, and independently optimize and
execute its ten-action suffix via the same 50-sample bounded CEM and common
root seed. Save RGB histories, executed actions, optimizer proposals/scores,
and cp2/cp7/cp17 physical position replay. Simulator position is evaluation
only; do not feed it into candidate generation or selection.

Report the model and physical nonprogress predicates separately, duplicate
action chains, physical results relative to zero/greedy, gradients/forward
samples, JVP provenance, encodings and physical steps. The common budget for
initialization and suffix planning does not certify equal total compute:
spectral generation uses a saved Jacobian, baseline prefixes are fixed, and
batch numerical work is shared. No common-suffix mechanism, native success,
independent roots, train-out-of-sample guarantee or admission is implied.
