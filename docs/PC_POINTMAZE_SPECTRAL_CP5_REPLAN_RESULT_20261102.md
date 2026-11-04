# PointMaze spectral prefixes with actual RGB tail replanning

The predeclared development protocol
`PC_POINTMAZE_SPECTRAL_CP5_REPLAN_PROTOCOL_20261102.md` completed on host
CUDA and the native RGB bridge. All 12 previously selected nonzero-control
prefix slots reproduced the sealed A1 step-5 RGB hashes. Each of four
prefixes then received the same eight-candidate, five-step tail search using
its own real step-5 RGB. The selected tail was executed in the same
environment to step 10.

Position-only distance to the archive frame-10 position, measured by a
separate offline physical replay (lower is better):

| Exposed root | Spectral | Random direction | Axis | Zero prefix |
| --- | ---: | ---: | ---: | ---: |
| 1880 | 1.70698 | 0.22512 | 0.66785 | 0.66786 |
| 1881 | 0.08579 | 0.46304 | 0.13531 | 0.12704 |
| 1882 | 1.08321 | 1.12152 | 1.76764 | 0.50976 |
| 1883 | 0.33253 | 0.33419 | 0.32862 | 0.12467 |

The spectral prefix is closer than zero on one of four roots and closer
than each of the random and axis prefixes on two of four roots. These are
four previously exposed development roots, not independent confirmation;
the distances are not native task success rates. This fixed exploratory
comparison does not establish a spectral advantage.

All tails received the same number of candidates and the same real-feedback
opportunity, but **total planning budgets were not equal**: source spectral
prefix construction used two JVPs and an SVD, the three source prefix
selectors each scored five actions, and zero was fixed. The original prefix
costs are carried as separate counts; the run cannot retrospectively
certify full equal-resource accounting. Root and per-branch elapsed times
are not interchangeable with total historical costs.

The planner input is actual RGB plus a constant raw-zero proprio vector.
It uses a single-frame context rather than an authenticated full action
history, and the checkpoint expects proprio; it is therefore a missing-input
ablation, not a fully authenticated RGB-only deployment. The model's
action normalization includes the full archive and its training provenance
is unknown. The camera was previously calibrated using an exposed root.
Physical goal position and endpoints were used only after all four branch
choices on a root were fixed; the separate replay evaluator is position-only,
and this run did not directly verify that the step-10 state of the RGB bridge
matched the replay backend for these newly chosen actions.

The result stores all goal/step-5/step-10 RGB images and pooled encoded
latents in the accompanying array archive, as well as candidate scores,
chosen actions, source hashes, and replay endpoints in JSON. Three tests
passed: validation of bounded/finite actions, finite deterministic
selection, and saved-artifact checks including hashes, source prefix/RGB
binding, argmin, and endpoint distance arithmetic. These tests do not
independently re-run model predictions or certify the physical backend.

- Runner: `le-wm-main/experiments/a1_pointmaze_spectral_cp5_replan_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json`
- Result SHA-256:
  `8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446`
- Test: `le-wm-main/tests/test_a1_pointmaze_spectral_cp5_replan_20261102.py`

This run does not establish a non-greedy reconfiguration mechanism,
equal-budget advantage, independent generalization, or Planning Compiler
P1. Its next useful checks include aligning the checkpoint's actual
observation/history contract and independently replaying the new actions
across the physical backends before interpreting the endpoint differences.
