# v22 PointMaze live-feedback development result

The preregistered v2 diagnostic ran both v22 response and multistart CEM for
three real RGB decisions on one setup, with horizons 15/10/5 and 1,920
prediction tokens per decision. The v1 attempt failed during renderer startup
before any planning or dispatch. Its manifest is retained. A separate setup
check showed the legacy renderer works with `DISPLAY=:0` and numeric GPU
device 0; v2 used that environment with a new output namespace.

The decision/data audit checked hashes, context history, executed actions,
float32 conversion, carry rescoring and prediction counts. The same-backend
RGB replay matched all 31 frames in each arm. A separate native physics replay
produced distances to the fixed target at the post-warmup boundary and after
each five-action segment:

| arm | initial | step 5 | step 10 | step 15 |
| --- | ---: | ---: | ---: | ---: |
| response | 1.720470 | 1.510756 | 0.814750 | 0.079619 |
| CEM | 1.720470 | 1.786326 | 1.362453 | 0.188421 |

These are replay distances, not a claim of independent evaluation, a
statistically meaningful advantage, or a certified task-success rate.
The v2 data audit reports `all_scored_min=false` because the same actual
float32 chain was scored more than once with slightly different values. The
separate score audit establishes that the minimum observed full-chain score
in each of the six decisions corresponds to the selected *action chain*.
The numerical drift's cause has not been isolated; the audit does not establish
model-prediction reproducibility across batch sizes.

An explicit-packet reference independently replayed all saved prediction
batches for both arms. Maximum absolute position error was 1.91e-6 and
maximum score error 1.67e-6, within the preregistered 2e-5 and 2e-6 limits.
This is same-checkpoint implementation consistency, not proof of physical
model accuracy or an independent replication.

Artifacts, all under `le-wm-main/results/`:

- `pc_v22_pointmaze_live_feedback_20261104_codex_manifest.json`: failed v1 attempt.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_manifest.json`: v2 dependency seal.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_{response,cem}_decision{0,1,2}.json`,
  corresponding image archives and execution receipts: full traces and dispatch.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_audit.json`: record/ledger audit.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_physical.json`: native replay.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_rgb_replay.json`: same-bridge image replay.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_score_audit.json`: scored-action audit.
- `pc_v22_pointmaze_live_feedback_v2_20261104_codex_{response,cem}_prediction_replay.json`:
  explicit-packet replay.

The goal and calibration were already exposed, source independence remains
unknown, total cost is not certified, and the physical replay shares the
backend. This development integration does not pass PC3, PC4 or PC5. The
method's PC2 evidence is unchanged.
