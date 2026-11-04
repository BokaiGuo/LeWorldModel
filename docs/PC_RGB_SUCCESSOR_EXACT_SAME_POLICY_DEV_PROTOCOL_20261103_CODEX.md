# RGB successor policy on exact wall dynamics: development protocol

This is a cross-backend engineering and PC2 development diagnostic, not an
independent confirmation and not a reopening of the stopped six-dimensional
block-program experiment. It runs the *unchanged* first-decision policy from
`pc_successor_rgb_execution_dev_20261103_codex.py` on the complete 24-root
wall development panel defined in
`PC_BLOCK_PROGRAM_WALL_PROSPECTIVE_DEV_PROTOCOL_20261103_CODEX.md`.
These roots have already been exposed. No root is selected based on outcome.

The exact dynamics accept normalized actions and expose endpoint position and
Euclidean terminal distance to the policy. A batch of n trajectories of
length 5, 10 or 15 costs n times length/5 prediction tokens; cap 480, including
the final zero-prefix counterfactual. Use the actual successor arm's fixed
nominal, seed, SVD, suffix allocation, joint refinement, and abstention
unchanged. SVD and execution time are reported separately; they are not
converted into tokens. Exact-state inputs are not legal visual-model inputs.

Seal every selected action chain, first candidate bank, all requested and
float32 model actions, positions, scores, budget counters, and source hashes
before any intervention. On the actual final chain, compare selected, zero,
and same-first-bank greedy prefixes. Greedy is chosen using exact 5-step
distance, with stable first-index tie breaking; this diagnostic evaluation is
outside the planner budget. For each prefix evaluate the same six ten-step
suffixes: zero, constant `(0.5,0)`, `(-0.5,0)`, `(0.5,0.5)`, `(0.5,-0.5)`,
and the actual selected suffix. Report absolute final distances and changes
from each prefix's own zero suffix. Independently run 120 full-chain samples
of the same 12-batch CEM per prefix using a common root-specific seed.

Record three predeclared diagnostic predicates: non-immediate progress
(`prefix_distance >= initial_distance - 1e-10`), selected actual suffix
relative change strictly below both controls by `1e-10`, and own replanning
terminal distance strictly below both controls by `1e-10`. All three must
hold on the same actual final chain. They are a limited development mechanism
screen, not a complete PC2 gate: random bridge, ordinary lookahead, and
additional passive-motion analysis remain necessary. Report all 24 roots,
success `<0.05`, mean distance, cost, conditions and failures. Do not tune
roots, nominal, budget, seed or conditions from the observed results.

This is not an independent strength comparison, not PC3, not evidence of
spectral increment and not an RGB transfer result. Any negative result
triggers action/interface and response-scale diagnostics before interpretation;
do not automatically attribute it to theory.
