# P1 continuation — implemented changes and current evidence

P1 **not passed**. No gate, native task evaluator, root-independence claim or frozen
checkpoint was changed. Existing sealed results preserved.

1. Completed checking the four Q-return arms: structured success at20, others
   unsuccessful. Its prefix improves jointXY, so this is not non-greedy proof.
2. Implemented equal-six candidate optimization: eight candidates × six steps,
   instead of eight × two plus32 winner-only steps. Four host-CUDA/physical arms
   completed, success0/4; all16 nodes verified. No initial prefix changed.
3. Located and repaired optimizer/replacement horizon mismatch in a separate
   variant. Jacobian, optimization and replay now use H3/H2/H1/H1, matching carry
   comparison. Four more host-CUDA/physical arms completed, success0/4; all16 nodes
   verified. Repair is real; causal sufficiency for P1 failure is not established.
4. Executed all eight fixed structured prefixes with own/shared/zero tails:
   24 new physical branches,14 complete and10 censored; no successes. Validated304
   image occurrences and encoded129 distinct images on host CUDA.
5. Same-action diagnostic retains both model/RGB ordering errors (2/6 comparable
   pairs) and RGB/task ordering errors (3/6). All pairs share one development root.
6. Completed actual-RGB suffix replanning for all eight fixed candidate prefixes:
   six complete20, two domain failures at4/5, success0/8. Checked24 full decision
   nodes and225 image occurrences; separately reproduced both domain failures.
   These forced-index interventions do not count as autonomous model choices.

This continuation adds40 physical branches, not40 independent tasks. RGB neural
work was CUDA only. Numerical fixture tests and physical simulation are not CPU
model fallbacks. Node counters exclude separate RGB/metric-construction costs;
no claim of full-cost superiority over a certified external baseline.

Details:
- `D2P_P1_Q_RETURN_RESULT_20261101_MAIN.md`
- `D2P_P1_EQUAL_CANDIDATE_Q_RESULT_20261101_MAIN.md`
- `D2P_P1_HORIZON_Q_RESULT_20261101_MAIN.md`
- `D2P_P1_CANDIDATE_INTERVENTION_RESULT_20261101_MAIN.md`
- `D2P_P1_CANDIDATE_FEEDBACK_RESULT_20261101_MAIN.md`

The fixed all-candidate own-suffix real-cp5 experiment is now complete, not pending.
Its results do not provide a passing action chain. Next candidate/search/metric
design must be explicitly frozen before new outcomes. No old outcome
may be relabeled independent confirmation. Method freezing, source-qualified new
roots, common-tail capability intervention and fair strong baselines remain
necessary before opening the scientific P1 gate.

Thirteen targeted numerical/contract regression tests passed; this does not
replace neural or scientific validation. The full16-slot algorithm comparison is
`le-wm-main/results/p1_return_variants_analysis_20261101_main_MAIN03.json`,
with explicit planner-only accounting scope.
