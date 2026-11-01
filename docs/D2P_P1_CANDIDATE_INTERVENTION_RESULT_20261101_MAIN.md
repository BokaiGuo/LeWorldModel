# Full candidate intervention and actual RGB diagnosis

## Completed evidence

- Fixed eight candidate prefixes × own/common/zero ten-action tails:24 physical
  branches. Fourteen completed15; ten were censored by the unchanged domain guard.
  None reached native success. Censored endpoints are not imputed.
- Saved consumer checked actual dispatch/return actions, state/frame alignment,
  repeated cp5 dynamics/RGB, native censorship and304 image occurrences.
- Frozen host CUDA encoded129 distinct pixel arrays covering all304 occurrences.
  No model updates. This is offline diagnosis, not extra feedback to a planner.

## What changed in the diagnosis

| Candidate | Prefix jointXY change | Own-tail final jointXY | Shared-tail final jointXY | Zero-tail final jointXY |
| --- | --- | --- | --- | --- |
| 0 | +0.00601 | 101.73380 | 109.74786 | 101.68241 |
| 1 | +5.14926 | 104.40471 | 130.61685 | 106.70910 |
| 3 | +2.14073 | 317.26044 | 43.23332 | 106.18669 |
| 7 (model choice) | −10.66530 | 22.95709 | 22.95709 | 91.39972 |

Other candidates have censored own/common branches. The full24-row JSON retains
them, including their stopping times. Candidate0 is the zero-action prefix;
small pusher motion is observed physical drift, not evidence of reconfiguration.

Candidate3 is a useful **diagnostic**, not a new chosen method: its prefix increases
jointXY but already decreases objectXY by58.17276 and angle error by0.47374.
Thus whether it is non-greedy depends on the predefined task metric; object-only
non-progress is NOT established. Its common-tail effect does not prove greater
planning capacity, and its own optimized tail performs poorly.

Candidate7 is best among the four uncensored own-tail endpoints under jointXY;
there is no observed hidden successful own-tail candidate to substitute for it.
The fixed bank therefore does not supply a passing P1 action chain.

## Prediction and metric are still separate failures

Only four own-tail endpoints are uncensored, yielding six correlated pairs on
one known root. Two pairs reverse predicted-Q versus actual-RGB-Q ordering;
three reverse actual-RGB-Q versus jointXY ordering. Not independent samples.

Example: candidate3's actual-RGB Q score0.15458 is better than candidate0's0.41921,
while jointXY317.26044 is far worse than101.73380. Goal-local Q therefore does not
solve global task-metric alignment. A model error and metric error can cancel;
do not interpret a physically correct ranking as a calibrated intermediate model.

## Sources and boundary

- `le-wm-main/results/p1_candidate_intervention_20261101_main_run_MAIN01/report.json`
- `le-wm-main/results/p1_candidate_intervention_analysis_20261101_main_MAIN01.json`
- `le-wm-main/results/p1_candidate_rgb_cuda_20261101_main_MAIN01.json`

P1 remains closed. No branch was promoted based on true outcomes. The next
discriminating experiment is an all-candidate, actual-cp5-RGB refreshed own-suffix
comparison with the same optimizer/resource contract, retaining censored prefixes.
That would test usable planning capacity; it must not be replaced by executing
only candidate3 or by retrospectively choosing its successful-looking shared tail.

Follow-up: the all-eight actual-cp5 experiment has since completed. See
`D2P_P1_CANDIDATE_FEEDBACK_RESULT_20261101_MAIN.md`; it remains unsuccessful0/8.
