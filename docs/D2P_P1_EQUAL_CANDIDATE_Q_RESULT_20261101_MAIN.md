# Equal-six allocation result

Four predefined arms each completed20 physical actions; native success0/4.
All16 saved nodes passed candidate-source, selected-plan/action identity, feedback
binding and ledger/model-hook accounting checks. No winner-only refinement events.
This does not independently replay all physics.

| Arm | Q winner-refinement success | Equal-six success | Equal-six final jointXY |
| --- | --- | --- | --- |
| structured | yes, step20 | no | 21.16242 |
| native | no | no | 68.30381 |
| random | no | no | 158.37768 |
| immediate_progress | no | no | 60.25019 |

All four initial selected indices and actual first-five-action prefixes remain
unchanged relative to Q winner refinement. Therefore this allocation change did
not uncover a new non-greedy prefix on this task. Structured retains immediate
jointXY improvement10.66530. Do not infer that equal allocation never helps.

The intended48 optimization updates are preserved for the first three arms;
immediate_progress increases16→48. Candidate final replay count falls9→8 for
the first three arms. Jacobian construction still differs across arms.
No equal-total-cost or independent advantage claim.

Machine-readable12-slot comparison:
`le-wm-main/results/p1_return_variants_analysis_20261101_main_MAIN01.json`.
All four receipts and CHECK files use
`p1_equal_candidate_q_20261101_main_{arm}_RUN_MAIN01`.
P1 remains closed. Next separately fixed experiment aligns optimizer and carry
comparison horizons; its protocol predates its physical results.
