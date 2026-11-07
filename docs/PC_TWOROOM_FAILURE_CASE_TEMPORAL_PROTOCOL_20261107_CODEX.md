# TwoRoom failure-case temporal decomposition

Retrospective diagnostic on the two previously exposed tasks 38 and 39,
selected because all original arms failed and fixed B banks contained
successes. For each, study the 64 sealed B actions and the actual previous
lookahead-selected chain. Do not interpret selection of these cases as
independent or representative.

First seal the frozen RGB-only model's initial and five predicted successor
embeddings for every chain, comparing final B costs against the earlier
whole-bank model scores. Then reexecute every action from the same physical
state, render at steps 5, 10, 15, 20 and 25, and check every B distance and
termination against the independently saved prior replay. Check prior
lookahead action identity and outcomes. Record initial RGB mismatch.

Finally encode real RGB observations and the exact goal image with the same
frozen encoder. Save predicted and real embeddings, goal embeddings, physical
distance and success at every step. Compare model and true-encoded scores,
feature discrepancy and physical outcomes at corresponding times, especially
the model-best B, successful lowest-score B and original selected chain.
The successful B identity is a retrospective oracle, never planner input.
Raw trajectories and hashes are retained. No algorithm is changed, no
statistical claim or gate pass is inferred from these two selected roots.
