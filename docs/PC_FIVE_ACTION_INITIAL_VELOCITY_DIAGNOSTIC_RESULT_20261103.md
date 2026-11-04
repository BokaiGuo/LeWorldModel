# Initial velocity diagnostic on the fixed five-action bank

The fixed-action diagnostic re-evaluated all 36 exposed candidate programs
under the historical initial zero-velocity input and, separately, the
simulator reset's instantaneous qvel. Qvel was used only for this offline
diagnostic; it is unavailable to the RGB-only planner. Original zero-input
predictions and actual cp5 RGB-derived scores reproduced their saved values.
No physical actions or optimizer calls were added.

| Root | Zero vs actual RGB-derived score | Qvel vs actual RGB-derived score | Zero vs physical distance | Qvel vs physical distance |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 5 / 36 | 2 / 36 | 6 / 36 | 1 / 36 |
| 8345 | 7 / 36 | 5 / 36 | 6 / 36 | 4 / 36 |
| 8349 | 8 / 36 | 6 / 36 | 7 / 36 | 5 / 36 |
| 8351 | 2 / 36 | 6 / 36 | 2 / 36 | 6 / 36 |

Entries are within-root pairwise ranking reversals, not independent trials.
The privileged input reduces aggregate reversals on this panel, but worsens
both comparisons on root 8351. It does not guarantee a better planner.

On root 8345, reset qvel was approximately `[-1.5453, -0.0698]`. For zero
versus bank index 5, the historical predicted scores were 2.0093 and 2.0079,
while qvel-conditioned predictions were 2.0227 and 2.0444; the actual
RGB-derived scores were 2.0710 and 2.0950, and physical distances were
2.0511 and 2.0848. The privileged diagnostic changes this pair's ordering
toward the actual ordering, but absolute errors remain. It does not isolate
all causes of the position-vector discrepancy in the original experiment.

The source bank directions were generated under zero initial velocity and
are held fixed in this comparison. Qvel is instantaneous; cp5 RGB velocity
is a temporal estimate, and full observation/velocity timing is not
certified. This retrospective analysis cannot be used to select an action,
certify deployable RGB feedback, establish causality, or open a gate.

The run used 45 encodes and 72 prediction samples; there were no new physical
steps. Full per-candidate scores, actual values, model hashes and input
bindings are in
`le-wm-main/results/pc_five_action_initial_velocity_diagnostic_20261103_DEV01.json`;
the pre-run protocol is
`docs/PC_FIVE_ACTION_INITIAL_VELOCITY_DIAGNOSTIC_PROTOCOL_20261103.md`.
