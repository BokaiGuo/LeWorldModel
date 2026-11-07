# Fixed failed-prefix RGB feedback: selected-case diagnostic

Roots 38 and 39 were previously exposed and selected retrospectively because
the actual initial lookahead-selected 25-step chain failed. Two independently
reset arms execute exactly its same first five actions (stop at success or
truncation). Thereafter:

* open-loop executes the unchanged sealed remainder;
* RGB feedback replans every five steps with the existing 680-query
  block-search baseline, remaining 25-step deadline, frozen model, newly
  rendered current RGB and goal RGB, and a re-scored old tail. Candidate
  seeds are fixed at 27121000 + 10 * root + iteration; planning begins at
  iteration 1. No action depends on evaluator-only state or success labels.

Record initial and boundary image digests, complete dispatched actions,
old-tail hashes, selection, candidate score, model query and packet counts,
GPU-synchronized planning time, task wall time, and physical success/distance.
Check both arms have the exact original first five actions and reproduce the
previous open-loop physical trace. Use exclusive output and bind code, weights,
input and prior artifacts. Include pre-existing 13/300-candidate RGB feedback
CEM results separately as reference, not as a matched causal control.

This tests whether this one fixed unsuccessful first plan can be corrected on
two chosen cases, not overall MPC superiority or descriptor value. The
feedback arm incurs extra queries and a changed execution path; even success
cannot establish same-compute improvement, generalization, or gate passage.
