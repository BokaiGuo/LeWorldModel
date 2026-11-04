# PointMaze fixed-proprioception diagnostic

Four known archive roots were scored on host CUDA with identical goal/root
proprio inputs fixed to one dataset-wide constant, using the same RGB, frozen
checkpoint, action bank and two-packet model scoring as the previously saved
RGB-plus-proprio run. Outcomes below are position-only fixed-bank
counterfactuals from the prior physical backend, not new physical executions.

| Root | Constant-input pair | Constant-input regret | Original proprio pair | Original regret |
| --- | --- | ---: | --- | ---: |
| 1880 | (2, 3) | 0.636081 | (4, 4) | 0.157374 |
| 1881 | (0, 0) | 0.192195 | (2, 3) | 0.342323 |
| 1882 | (3, 3) | 0.284369 | (3, 6) | 0.141739 |
| 1883 | (2, 0) | 0.065061 | (7, 4) | 0.038427 |

All four choices changed; three regrets increased and one decreased. This is
a tiny known-root distribution-shift diagnostic. It does not isolate model
prediction from task-metric mismatch or prove an online RGB-only planner.

## Important normalization source correction

The saved report says global normalization provenance is unknown. Inspection
of `external/jepa-wms/app/plan_common/datasets/point_maze_dset.py` shows the
loader derives `proprio_mean` from `self.states.clone()` across every loaded
trajectory, and derives action statistics from all loaded trajectories too.
This run loaded all 2,000 trajectories, including the four evaluation roots.
Consequently the constant contains aggregate evaluation-trajectory state
information and action normalization likewise includes evaluation episodes.
No root-specific proprio was supplied to the planner, but this is **not an
independent-normalization or clean RGB-only confirmation**. Checkpoint
training provenance is still unknown. Preserve the raw result as a
development artifact; do not use it as a formal independent outcome.

The saved-array validator recomputed every 8x8 physical distance matrix,
argmin and selected regret from the source positions, verified SHA binding
and frozen parameter equality, and passed for all roots. This is not a new
physical replay or a complete no-leak certification. Five tests passed,
including explicit state-independent constant construction and rejection of
invalid saved choices. Fixed preprocessing statistics from an authenticated
training-only source are required for the next confirmatory experiment.

Artifacts:

- `le-wm-main/results/a0_pointmaze_constant_proprio_20261102_DEV01.json`
- `le-wm-main/results/a0_pointmaze_constant_proprio_validate_20261102_DEV01.json`
- `le-wm-main/experiments/a0_pointmaze_constant_proprio_20261102.py`
- `le-wm-main/experiments/a0_pointmaze_constant_proprio_validate_20261102.py`
- `docs/PC_POINTMAZE_CONSTANT_PROPRIO_PROTOCOL_20261102.md`
