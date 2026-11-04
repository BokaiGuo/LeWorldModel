# Goal-mode RGB MPC vs direct CEM: one-root exploratory comparison

Both methods used the same real warmup, frozen model, initial root, three
five-action execution/feedback intervals and 132 prediction-token cap per
decision. Both used 15-action model plans and executed only the first five
actions before refreshing RGB. Direct CEM planned 44 complete trajectories
per decision. The goal-mode run additionally computed a finite-difference
response and an SVD; equal prediction tokens therefore do not certify equal
total compute or compensate CEM for extra structural overhead.

| Distance to goal | After warmup | After 5 | After 10 | After 15 |
| --- | ---: | ---: | ---: | ---: |
| Goal-mode with zero-prefix abstention | 1.745588 | 1.471325 | 1.131087 | 0.814188 |
| Direct CEM | 1.745588 | 1.748452 | 1.433446 | 0.964716 |

Distances are from separate fixed-action post-seal MuJoCo 2.1.2.14 replays.
The baseline protocol was written after the goal-mode outcome was already
observed, and both use one previously used development root. The result is
not independent, has no uncertainty estimate, is not a fairness-certified
budget comparison, and cannot establish PC2 or PC3. It does not isolate the
value of goal-ranked modes from ordinary candidate generation or lookahead.
Further development requires same-bank, no-spectral and random controls,
cost-complete resource policy, and provenance-qualified independent roots.

Artifacts:

* `le-wm-main/results/pc_goal_mode_abstain_rgb_mpc_20261103_codex_physical.json`
* `le-wm-main/results/pc_goal_mode_rgb_mpc_cem_20261103_codex_physical.json`
