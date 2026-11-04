# Full-CEM tail carryover under real RGB feedback

On the two previously observed Reacher development roots, retain the original
full-CEM first packet and its two-packet pre-feedback tail. After executing the
first packet, evaluate the old tail as candidate 1 in each iteration of the
same successor CEM (16 candidates, 2 iterations, 4 elites, root+5000).
Candidate 0 remains the current mean; the remaining 14 candidates come from
the original seeded random draw. Choose the lowest model cost from the final
iteration, as in the existing successor CEM. The old tail displaces a random
candidate, so this adds no model rollout evaluations. It does not give the
method a guarantee relative to the original sampled population.

Use the original frozen checkpoint, root and goal RGB hashes, action history
and first-packet witness. Execute the selected suffix in the real environment.
Bind source reports and model hashes; log old-tail and chosen predicted costs,
whether old tail was selected, raw actions, physical outcomes and per-class
call counts. The initial three-packet CEM cost is inherited from the sealed
original run; initial-search wall time is not measured by this execution.
The original comparison is retained. These are known development roots,
observed before this protocol; results cannot establish independence,
equal-compute fairness or a P1/PC gate.
