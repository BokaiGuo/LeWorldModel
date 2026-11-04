# CP12 single/three-frame projected task-score rankings

The prewritten `PC_POINTMAZE_CP12_HISTORY_TASK_RANK_PROTOCOL_20261103_CODEX.md`
was executed in two separate processes: first seal scores computed from the
32 saved predictions and RGB goal; then open the old cp12 refresh receipt
for actual RGB-derived scores and evaluation-only physical distances.
No candidate was selected and no physical state was supplied to a planner.

Strict within-root pair reversals; lower means greater agreement with that
reference, not better demonstrated control:

| Root | One vs actual RGB | Three vs actual RGB | One vs physical distance | Three vs physical distance |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 2/28 | 3/28 | 3/28 | 4/28 |
| 8345 | 5/27 | 4/27 | 6/28 | 5/28 |
| 8349 | 0/28 | 1/28 | 0/28 | 1/28 |
| 8351 | 1/28 | 1/28 | 2/28 | 2/28 |
| Sum | 8/111 | 9/111 | 11/112 | 12/112 |

Root 8345 has one tie in the actual RGB score, excluded from the 27
comparable pairs. There are no prediction or physical-distance ties. The
actual RGB ranking reference is the original saved refresh score; the new
score computed from saved cp17 RGB-derived proprio tokens is checked against
it rather than substituted, preserving the prior receipt's tie definition.

Single-frame scores reproduce all 32 old refresh scores within the frozen
`rtol=3e-5, atol=1e-6`; maximum absolute difference is `1.66893e-6`.
Recomputed actual RGB task scores match their old receipt values within the
same tolerance, maximum absolute difference `1.90735e-6`. All four single
arm ranking and tie counts exactly match the old refresh receipt. The goal
token was reconstructed on CPU from the checkpoint's affine proprio encoder
and the same calibrated RGB goal state. The reproduction is tolerance-based,
not bitwise; CPU/GPU arithmetic and analytic token reconstruction are
plausible sources of the small differences, not independently isolated.

Compared with one frame, the three-frame score has fewer reversals for 8345,
more for 8343 and 8349, and the same counts for 8351. This exposed four-root
development panel does not establish that three-frame history improves or
worsens general ranking, selection, or physical control. Pair counts are
dependent and candidates were previously exposed and selected; they are not
independent trials. The old RGB state estimation, render timing, checkpoint
training provenance and physical replay limitations still apply. The goal
and predictions are RGB-derived; privileged physical distance is evaluation
only. No new model or physical experiments were performed.

New budget: 96 projected score evaluations, one analytic affine goal token,
zero predictor calls, zero image encodes, zero physical steps and zero
candidate selections. The inherited same-action run used 32 single- and 32
three-frame predictor samples and 128 observation encodes; three-frame
conditioning uses longer context and two additional observations per branch.
This analysis is not an equal-compute planner comparison.

- Score receipt SHA-256
  `0ae7558da21212e8b97d79d35b46e05fba3c9243eaf0acef9032f4133b1eee9b`.
- Evaluation receipt SHA-256
  `baa8b7da6d3432b9eb4658f56f8a49d6d8e56bdb3854ab9a19f72521a9c0a009`.
- Focused tests: **8 passed**, including independent NumPy recomputation of
  the 96 saved scores and explicit pairwise ranking loops. Tests consume
  saved output and do not rerun inference or certify the complete transitive
  dependency closure.
