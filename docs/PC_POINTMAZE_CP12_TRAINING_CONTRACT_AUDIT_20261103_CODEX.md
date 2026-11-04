# PointMaze cp12 training/runtime contract audit

## Scope and result

This is a read-only, development audit of the already executed cp12
refresh, real-feedback replan, matching controls and imagined-feedback runs.
It checks the JEPA-WMs PointMaze loader and wrapper used by those runs, not
the separate repository-root LeWM training code. No old script, checkpoint,
configuration or result was changed. No neural inference or physical replay
was performed in this audit.

**No concrete action normalization, packet-order or off-by-one bug in the
recorded cp12 paths was established.** Two material qualifications remain:
the runtime intentionally uses a single encoded observation where the
inspected training configuration uses a three-frame history, and the cp12
receipts do not independently bind every dataset statistic or imported
dependency at execution time.

## Action normalization and provenance

- The inspected `PointMazeDataset` computes per-coordinate means and sample
  standard deviations over each valid trajectory before the train/validation
  split and normalizes each raw 2-D action once
  (`external/jepa-wms/app/plan_common/datasets/point_maze_dset.py:30–73`,
  `datasets/utils.py:164–181`). The current runtime loader uses this same
  dataset with `normalize_action=True`.
- Running that actual loader over the current 2,000-trajectory archive gives
  mean `[0.0000782133866, 0.0006003659801]` and std
  `[0.5769129992, 0.5775879025]`. All action and proprio means/stds equal
  the local JEPA-WMs `DATA_STATS["pointmaze"]` bit-for-bit; the registry
  asserts its values match training. The action values also equal the older
  saved archive-alignment receipt, whose archive files are hash checked.
  This is stronger than a mere self-consistent round trip, but the registry's
  assertion is not an independent checkpoint-specific training receipt.
- The cp12 scripts validate finite positive action std and check proprio
  statistics, but do **not** record and bind their actual action mean/std or
  `actions.pth` hash in the cp12 result. The checkpoint parameter hash covers
  named parameters, not these externally computed preprocessing statistics.
  Thus current archive equality plus an older receipt does not independently
  prove the statistics were unchanged at the historical cp12 execution.

## Packet, clock and feedback

- The inspected configuration specifies `frameskip=5`, `action_skip=1`,
  `num_hist=3`, `num_pred=1`. The actual training slicer takes observations
  at `t,t+5,...` and flattens the five chronological 2-D actions beginning
  at each observation into one time-major 10-D packet
  (`datasets/traj_dset.py:130–154`). An independent test exercises the actual
  slicer with distinguishable action coordinates and timestamps.
- For cp12, the real-feedback runner replays raw actions `[0:12]`, estimates
  the state from images 11 and 12, normalizes actions `[12:17]` once and
  reshapes to `[1,1,10]`. It predicts one packet and decodes/executes five
  raw commands. The imagined arm predicts cp7→cp12 from `[7:12]`, then
  replans `[12:17]` (`pc_causal_mixed_cp12_replan_20261103.py:73–105`;
  `pc_causal_mixed_cp12_imagined_20261103.py:109–145`). These slices match
  the stated command clock. The five MuJoCo integration ticks **per raw
  command** in `pointmaze_native_rgb_bridge_20261102.py:60–69` are distinct
  from the five raw commands per model packet.
- Across 32 real-replan, 8 real-control and 40 imagined branches, the test
  checks original first-12 actions and first-13 saved images, the saved final
  CEM score winner, and exact float32 denormalization/clipping into saved
  executed actions. These checks validate recorded array bindings, not
  historical hardware execution or neural predictions.
- The loader explicitly passes `ctxt_window=1`
  (`run_sra_p0_pointmaze_cuda.py:139–170`). Real feedback encodes one RGB
  frame plus an RGB-derived position/velocity estimate; imagined feedback
  uses one model-predicted visual/proprio state. The actual wrapper's
  single-frame autoregressive action/proprio order was tested with a
  synthetic predictor; existing CUDA single-frame parity is recorded in
  `PC_POINTMAZE_ACTION_HISTORY_UNROLL_VALIDATION_RESULT_20261103.md`.
  The inspected training config has `num_hist=3` and rollout context window
  3, so the cp12 path does not reproduce that history conditioning. This is
  a real, disclosed inference-contract difference, not proof that using a
  shorter context is an implementation error or that results would improve
  with history.
- `causal_state` divides successive image-derived positions by 0.05 s; this
  gives an interval-average estimate, not the instantaneous training-state
  velocity. The existing independent tick diagnostic found the current
  bridge's post-step render cached one integration tick behind boundary
  qpos, but could not reproduce historical saved frames byte-for-byte.
  Therefore it identifies a current-bridge risk, not a proven exact cp12
  historical timing error.

## CEM boundary risk, not an established cp12 failure

`pc_bounded_packet_cem_20261103.py:18–36` scores projected candidates but
updates elite mean/std from the unprojected proposals. If an elite lies
outside bounds, the update can differ from a CEM update on evaluated
actions. This is a code-level boundary-handling risk, not automatically an
incorrect algorithm for every intended clipped-Gaussian variant.

For the 80 saved replan/control/imagined branches, all 240 observable
successive CEM elite means (three iterations per branch) exactly equal the
mean reconstructed from the scored, projected saved elites; scores contain
no ties in these comparisons. This finds no mean discrepancy in the saved
runs. Raw pre-projection proposals and elite standard deviations were not
saved, and final/intermediate means can themselves be projected, so the
audit cannot exclude cancellation, hidden clipping, or a changed sampling
std. The replan receipt binds the inspected CEM kernel hash; the imagined
receipt does not itself record that kernel hash.

## Reproducibility and evidence boundaries

New test:
`le-wm-main/tests/test_pointmaze_cp12_training_contract_audit_20261103_codex.py`.
It pins current reviewed source hashes, source result/array hashes, cp12
runner hashes, checkpoint hash, config hash and relevant archive receipts;
unexpected changes fail rather than silently substituting inputs. The
reviewed external repository is at `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`;
its local inference wrapper is modified relative to that Git HEAD (the
multi-context action-feature padding change). The tested `ctxt_window=1`
path is conditional on the pinned current wrapper. The cp12 manifests do not
bind that wrapper's source hash, so the present local file is not by itself
proof of historical dependency bytes.

The independent test plus three existing focused cp12 test files: **10
passed**, on CPU:

```sh
MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=le-wm-main:le-wm-main/external/jepa-wms le-wm-main/.venv/bin/python -m pytest -q -p no:cacheprovider le-wm-main/tests/test_pointmaze_cp12_training_contract_audit_20261103_codex.py le-wm-main/tests/test_pc_causal_mixed_cp12_refresh_20261103.py le-wm-main/tests/test_pc_causal_mixed_cp12_replan_saved_audit_20261103.py le-wm-main/tests/test_pc_causal_mixed_cp12_imagined_20261103.py
```

These tests do not certify the checkpoint's original training config/data,
train/test independence, cp12 RGB-state accuracy, physical success, or
closed-loop scientific claims.
