# Extra-budget CEM: four-root feedback development result

The fixed follow-up ran the same four exposed roots and three real-RGB
feedback decisions as the matched-token panel, with two predeclared CEM
budgets. All eight trajectories were sealed and replayed on the same
MuJoCo 2.1.2.14 backend.

| Root | Goal mode, 132 tokens | CEM, 132 tokens | CEM, 144 tokens | CEM, 156 tokens |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.6291 | 0.5906 | 0.5906 | 0.3167 |
| 1 | 0.4990 | 0.5861 | 0.5861 | 0.5717 |
| 2 | 0.3683 | 0.5645 | 0.5645 | 0.5183 |
| 3 | 0.1417 | 0.5206 | 0.5206 | 0.4552 |
| Mean | 0.4095 | 0.5654 | 0.5654 | 0.4655 |

Entries are terminal physical distance, lower better. All four CEM-144
terminal distances equal the recorded CEM-132 values; CEM-156 has lower
terminal distance on all four roots and beats goal mode on root 0. Goal mode
has lower terminal distance on the other three roots. This does not establish
a reliable advantage: roots and method were previously used in development,
one optimizer seed per root was used, and the extra-token protocol followed
earlier observed results.

Mean recorded planning time per decision was 1.2354 s for goal-132 in the
earlier panel, 1.1786 s for CEM-132, 1.2975 s for CEM-144 and 1.3906 s for
CEM-156. These are separate-run, machine-specific wall-clock observations;
they are not an exchange rate between prediction tokens and SVD or a formal
equal-compute guarantee. Candidate generation, feedback and relevant
structural calls are reported separately. `full_cost_certified` remains false.
The previously tested same-bank greedy mean was 0.3482; this development
comparison does not demonstrate that goal-ranked selection adds value.

Separately, a post-outcome audit of the actually executed goal-mode chains
found 0/12 non-improving physical prefixes. That excludes those 12 chains
from satisfying the non-immediate necessary condition, but says nothing
about whether another task or method can produce a valid mechanism.
No PC gate is promoted.

Artifacts:

* `le-wm-main/results/pc_cem_extra_compute_feedback_20261103_codex_physical.json`
* `le-wm-main/results/pc_goal_mode_matched_feedback_20261103_codex_selected_audit.json`
