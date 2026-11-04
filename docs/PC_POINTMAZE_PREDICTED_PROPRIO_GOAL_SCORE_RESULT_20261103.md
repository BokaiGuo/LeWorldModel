# Predicted proprio goal scoring on fixed development banks

The [fixed protocol](PC_POINTMAZE_PREDICTED_PROPRIO_GOAL_SCORE_PROTOCOL_20261103.md)
was run on host CUDA with the frozen checkpoint. All 624 candidates in 12
previously exposed, outcome-selected banks were rescored. The goal state was
estimated from its observed RGB using the existing marker projection, with
zero goal velocity; no physical outcome or true future image was used to
compute either score. The saved original visual scores matched recomputation
at the preregistered tolerance. Model parameter hashes matched before/after.

Relative to the original visual-score choice, the predicted-proprio-score
choice has smaller sealed physical endpoint distance in 7 banks and larger
distance in 5. Its physical-ranking reversal count is smaller in 11 of 12
banks. Bank comparisons share three exposed roots, are not independent
replicates, and need not transfer to a reoptimized candidate distribution.

For the root 1893 random-extra tail bank, the visual and proprio choices
have physical distances 1.3201 and 0.46994 respectively; the latter is the
same-bank physical oracle. Its pairwise reversals against physical distance
are 807/1128 and 30/1128 respectively. Conversely, root 1895 random-extra
tail worsens from 0.20818 to 0.58465, with reversals rising from 205/1128
to 639/1128. Both examples are retrospective, not new executions.

This is an actionable alternative frozen-model score for subsequent
prospective development, not a demonstrated controller improvement: CEM
optimized with this score would generate a different bank, and goal velocity
zero may not be valid for every task. The source goal/state projections and
normalizer provenance remain bounded as previously recorded. Physical
distance is evaluation-only. No independent root, equal-budget baseline,
non-greedy reconfiguration mechanism, P1 gate or full scientific closure is
established.

- Report:
  `le-wm-main/results/a0_pointmaze_predicted_proprio_goal_score_20261103_DEV01.json`,
  SHA-256 `a3339218d9f5dc86da737fd773e30f65eaeefbcaf5f6849791b30cf251690d68`.
- Arrays: same basename `.npz`, SHA-256
  `7715f7cb9265cad7c4eb875b045ce853c917fe5dd466c529a9226738eb2a136c`.
- Code:
  `le-wm-main/experiments/a0_pointmaze_predicted_proprio_goal_score_20261103.py`.
- Tests: `tests/test_a0_pointmaze_predicted_proprio_goal_score_20261103.py`,
  **2 passed**, covering score reduction and saved outcome/ranking bindings;
  they do not rerun the checkpoint or simulator.
