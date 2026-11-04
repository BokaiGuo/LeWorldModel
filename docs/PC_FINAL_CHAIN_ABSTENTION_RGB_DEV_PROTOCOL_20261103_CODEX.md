# Final-chain abstention RGB development protocol

This is a new development policy, not the frozen full Planning Compiler and
not an independent confirmation. It reuses the fixed-terminal 15/10/5 RGB
feedback runner and the previously exposed four development roots. The
existing adaptive, uniform, greedy and CEM arms remain distinct.

At each decision reserve one full-horizon prediction from the 480-token cap
before invoking the existing traced policy. After all generation, refinement,
and carry selection, score the actual selected chain with its first five
actions replaced by zeros and its remaining suffix unchanged. If this
counterfactual is no worse, execute it and record abstention; otherwise
execute the original selected chain. Ties abstain. CEM reserves and spends
the same token amount rescoring its own chain, without an abstention claim.
All comparisons use the same current observation and target. The diagnostic
comparison, score, actual chosen chain and token charge must be saved.

This is a model-scored safety fallback, not proof of a reconfiguration
mechanism. It does not compare independently optimized suffixes and cannot
certify PC2. Equal prediction-token caps are not equal total compute:
encoder, differentiation, decomposition, simulation, and wall time must
remain separate. The existing runner declares `full_cost_certified=false`.
Historical results must not be overwritten or relabeled.

Development outcomes, including negative outcomes, must be recorded without
selecting a favorable root. This protocol cannot establish PC3 or independent
confirmation; its roots and checkpoint provenance do not satisfy those
requirements. No old PointMaze non-immediate mechanism protocol is reopened.
