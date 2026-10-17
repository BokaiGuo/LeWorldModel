# PointMaze external action-effect adjudication

This note freezes the claim-safe adjudication of the 2026-10-17 external
PointMaze experiments. It deliberately separates the untouched confirmatory
evaluation, the post-failure exploratory repair, and the online planner-ceiling
diagnostic.

## Confirmatory external result

The evaluation used 100 root-disjoint episodes (`1900--1999`), five adapter
seeds, a frozen action bank, and a calibration-only dataset-matched renderer.
The renderer is **not** untouched raw sensor RGB.

| Comparison | Regret gain | Paired 95% CI |
| --- | ---: | ---: |
| Bank-only | 0.03694 | [0.01556, 0.05867] |
| Shuffled prediction | 0.07758 | [0.05101, 0.10229] |
| No dynamics | 0.14081 | [0.11329, 0.16623] |
| Unseen candidates vs shuffled | 0.07472 | [0.05872, 0.09033] |
| Matched temporal order vs bank-only | 0.01385 | [0.00482, 0.02388] |

The five ranking controls are positive, but the frozen joint release criterion
does not pass. At epsilon 0.25, only 54/100 external roots pass support, for a
release rate of 0.54 versus the preregistered minimum of 0.75. The released
violation rate is 1/54 (0.0185), and the calibration bound is 0.10790.

**Adjudication:** the action-effect adapter has independent fixed-bank ranking
information, but confirmatory HTSC usefulness fails because deployment coverage
is too low. This is not a fully successful external validation.

## Root cause and exploratory repair

The main coverage failure is a representation-interface mismatch: development
used stored RGB while deployment used the calibrated renderer. The paired
latents remain nearly collinear, but small scale/offset changes are amplified by
the train-only standardization and PCA support metric.

Retraining the rank-32 adapter on development roots rendered through the same
calibration-only interface raises support release from 54/100 to 90/100. At
epsilon 0.25, the released violation rate is 2/90 (0.0222), effect RMSE is
0.47450, and outcome regret is 0.04399.

This repair was designed after inspecting the external support failure and
reuses the same roots. It is therefore **exploratory cause-and-repair evidence**,
not a second confirmatory result.

## Online planner-ceiling diagnostic

With 24 action blocks on 32 external roots, the exact candidate oracle succeeds
on 23/32 episodes, perfect-dynamics MPC on 22/32, and calibration-only visual
dynamics on 23/32. The frozen RGB world-model policy succeeds on only 2/32.

| Method | Success | Mean final distance |
| --- | ---: | ---: |
| Exact candidate oracle | 23/32 (71.88%) | 0.2994 |
| Perfect-dynamics MPC | 22/32 (68.75%) | 0.3144 |
| Visual dynamics | 23/32 (71.88%) | 0.3074 |
| Frozen WM RGB | 2/32 (6.25%) | 1.0371 |

The larger budget repairs the earlier planner ceiling. The remaining online
failure is therefore dominated by frozen world-model action ranking, rather
than by candidate-bank reachability alone.

## Current claim boundary

- Supported: root-disjoint fixed-bank action-effect ranking contains positive
  information beyond bank-only, shuffled, no-dynamics, unseen-candidate, and
  matched-order controls.
- Failed: the confirmatory useful-release conjunction because coverage is 54%.
- Exploratory: renderer-matched training repairs the measured support mismatch.
- Unresolved: integrating the leak-free action-effect ensemble into online RGB
  planning and validating it on a genuinely unopened environment or root set.
- Not supported: universal world modeling, a universal commutator certificate,
  repaired frozen predictor dynamics, or untouched raw-RGB planning.

The compact machine-readable companion is
`external_action_effect_adjudication_20261017.json`. Large per-root vectors and
traces remain in the ignored experiment directory and are not committed.
