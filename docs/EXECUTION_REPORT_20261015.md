# LeWorldModel execution report — 2026-10-15

This continuation used the project environment and host-permission CUDA. The
host GPU was checked before every learned replay: NVIDIA GeForce RTX 5060 Ti,
`torch 2.12.0+cu130`, and a real CUDA allocation succeeded. No run in this
report was silently rerouted to CPU.

## Code changes

`le-wm-main/experiments/run_rgb_local_effect_adapter.py` now has a
support-conditioned online local-effect path:

- local history is collected only across consecutive `torque/hold` frames;
- translation/recovery or missing RGB contact support clears the history;
- local angular correction requires finite action-design conditioning and a
  recent-fit angular residual below the frozen default threshold (`0.25 rad`);
- the correction fails closed to the frozen adapter when the gate is not met;
- trace fields expose the gate, residual, and condition number.

`replay_rgb_local_effect_planning.py` forwards the new protocol controls and
now handles newer support-constrained KNN decoder artifacts without treating
their tensors as affine calibrators. An explicit request to force an
unsupported optional decoder still fails closed.

Validation: `py_compile`, `git diff --check`, and 22 targeted tests pass.

## Same-artifact local-angle audit

All four replays use the same frozen paired adapter, terminal roots, action
bank, 4 episodes, 40 steps, and host CUDA. They differ only in the declared
angle interface or local-history intervention.

| Replay | Strict success | Mean XY (px) | Mean angle (rad) | Contact retention |
| --- | ---: | ---: | ---: | ---: |
| paired chart baseline | 0/4 | 178.996 | 1.639 | 0.881 |
| local ridge + learned angle | 0/4 | 217.225 | 1.395 | 0.888 |
| local ridge + RGB angle | 0/4 | 177.612 | 1.582 | 0.875 |
| low-support diagnostic (`min_samples=2`, window 8) | 0/4 | 183.298 | 1.575 | 0.894 |
| support-conditioned history + residual gate | 0/4 | 178.996 | 1.639 | 0.881 |

The learned-angle condition improves angle error but worsens translation. The
RGB-angle condition is nearly unchanged. The support-conditioned gate rejects
the local fit on every audited frame because the terminal protocol does not
provide a consecutive, full-rank, verifiable within-contact action history.
This is a support diagnosis, not evidence that the gate itself is ineffective.

Artifacts:

- `results/negative_cause_diagnosis_20261015/replay_paired_chart_native_cuda/`
- `results/negative_cause_diagnosis_20261015/replay_paired_local_angle_learned_cuda/`
- `results/negative_cause_diagnosis_20261015/replay_paired_local_angle_rgb_cuda/`
- `results/negative_cause_diagnosis_20261015/replay_paired_local_angle_rgb_support_diag_cuda/`
- `results/negative_cause_diagnosis_20261015/replay_paired_support_conditioned_rgb_cuda/`

## Support-root adapter intervention

A fresh GPU adapter was trained with `terminal_root_support_gap=16` and 64
support-angle samples. Its held-out action-effect RMSE was `4.2985`; contact
action top-1 was `0.1905`. Its same-run terminal planning reached contact
retention `0.9688` but strict success remained `0/4`.

The independent replay using the validation current-XY observer produced:

- strict success: `0/4`;
- mean XY: `160.555 px`;
- mean angle: `1.825 rad`;
- contact retention: `1.000`.

Artifact: `results/negative_cause_diagnosis_20261015/replay_support_roots_validation_cuda/`.

The intervention therefore repairs contact retention but not the downstream
decision consequence. The remaining bottleneck is within-contact
action-to-angle identifiability, not merely terminal support or passive
history length.

## Current evidence boundary

The positive result established in this continuation is an engineering and
safety result: invalid local identification is now observable and rejected,
and support-root construction can preserve contact. There is still no stable
pose closure, no independent strict visual-planning gain, and no evidence for
universal learned causal dynamics or a universal commutator certificate.

The next admissible experiment was an explicitly contact-mode- and
action-support-conditioned active probe, with probe success and real/predicted
successor consistency measured on root-disjoint held-out roots. Further global
angle-weight or memory-family tuning is not justified by the present evidence;
its execution is documented below.

## Active paired-probe support repair (2026-10-15)

The next repair was implemented and run on host CUDA.  The original online
cache could reuse a single zero-angle probe as if it were a causal local
effect, which caused stale action repetition.  The controller now requires
informative support on both sides of a matched moment contrast, forces an
alternating probe when that support is absent, and matches the opposite pulse
in the local normal/tangential action chart.  A stiction-aware maximum-moment
option and a two-step identification pulse were also added as declared
protocol controls.

The three-seed root-disjoint panel is
`results/negative_cause_diagnosis_20261015/active_paired_probe_panel_20261015/`.
It uses 24 train roots, 16 held-out roots per seed, the same boundary-contact
candidate bank, 64 trees, 12 closure steps, and CUDA bootstrap aggregation.
Across 48 held-out roots, adaptive paired probing versus global one-step gives:

| metric | global | adaptive paired | paired delta (95% CI) |
|---|---:|---:|---:|
| predicted/observed effect error | 0.1492 | 0.1014 | -0.04775 [-0.07267, -0.02157] |
| pose success | 4.17% | 4.17% | 0 [-0.0, 0.0] |
| physical boundary contact loss | 0 | 0 | 0 |
| final angle (rad) | 1.4628 | 1.5511 | +0.0883 [+0.0576, +0.1212] |

The intervention therefore solves the stale-cache and local action-effect
identification failures, with a reproducible effect-transfer improvement and
no physical contact-loss penalty.  It does not solve angle closure: the
adaptive arm is significantly worse on final angle in this protocol and has
no pose-success gain.  The remaining bottleneck is multi-step transfer from
identified local effects to task orientation, not absence of local excitation.
This artifact is mechanism-level evidence only; stable adapter closure,
frozen RGB planning, universal physical causality, and universal commutator
certification remain closed.

## Initial-angle support gate and phase-aware follow-up

The next protocol repair made the local orientation-support domain explicit in
`experiments/run_pusht_mode_aware_shared_candidate.py` through
`--max-initial-angle-error`.  A finite gate oversamples independent terminal
roots and keeps only roots inside the declared domain; it does not select roots
using any planning outcome.  The report now records the train/held-out angle
distribution and the complete action protocol.  The default remains infinite,
so legacy runs are not silently changed.

The first CUDA smoke used the boundary-contact support gate
`min_goal_support_alignment=0.35`, `max_initial_angle_error=1.55`, action
magnitudes `(0.10, 0.16, 0.24, 0.32)`, inward magnitude `0.12`, two-step
pulses, and no relocation.  It reached `2/8` strict pose success, with
boundary contact loss `0` and mode accuracy `0.9488`.  This was not promoted as
stable evidence because the panel was small and the gated roots are not paired
with the ungated roots.

The larger root-disjoint CUDA panel is
`results/negative_cause_diagnosis_20261015/initial_angle_gate_cuda_panel_32x16_seed20261027/`.
The held-out angle errors have mean `0.8648 rad` and maximum `1.5293 rad`.

| controller | strict pose | final XY (px) | final angle (rad) | boundary contact loss |
|---|---:|---:|---:|---:|
| global one-step | 3/16 (18.75%) | 39.11 | 0.544 | 0/16 |
| mode-aware lookahead | 2/16 (12.50%) | 51.51 | 0.463 | 0/16 |
| goal-conditioned ranker | 2/16 (12.50%) | 21.88 | 0.652 | 0/16 |

This isolates an important cause but not a repair: removing large initial angle
errors does not produce reliable closure.  The mode-aware controller reduces
angle error while losing translation, whereas the goal-conditioned ranker
recovers translation while leaving a larger angular residual.  The remaining
failure is therefore a within-contact translation/rotation transfer conflict,
not only an initial-angle-support problem.

The strict phase-only follow-up (same `32/16`, `64` trees, action bank and
support gate; no recovery branch) is
`results/negative_cause_diagnosis_20261015/initial_angle_gate_phase_only_cuda_panel_32x16_seed20261028/`.
It gives mode accuracy `0.9546`, physical contact loss `0`, and:

| controller | strict pose | final XY (px) | final angle (rad) |
|---|---:|---:|---:|
| global one-step | 2/16 (12.50%) | 43.68 | 0.472 |
| mode-aware lookahead | 3/16 (18.75%) | 39.03 | 0.508 |
| goal-conditioned ranker | 3/16 (18.75%) | 24.80 | 0.634 |

Phase-aware scoring lowers angular error in some arms but does not yield a
stable strict-pose gain.  An exploratory run with state-only goal-directed
recovery reached `7/24` for the goal-conditioned ranker, but it uses a
different `48/24` protocol and a privileged recovery interface; it is not
attributed to the learned controller.

The current evidence-supported diagnosis is now two-stage: task-compatible
contact and successor-mode prediction are recoverable, but the learned/local
action interface does not provide a stable multi-step within-contact mapping
that simultaneously preserves XY and closes orientation.  Stable pose closure,
frozen RGB online planning, universal learned physical causality, and a
universal commutator certificate remain closed.

## Structured two-stage control and candidate-support follow-up

The runner now contains a separately reported `staged` arm.  Its stage is
state-only and frozen at an XY error of `28 px`: outside the tube it uses the
goal-conditioned translation ranker, and inside the tube it uses the
mode-aware action-effect ranker.  The stage decision never reads a simulator
successor.  The implementation is in
`experiments/run_pusht_mode_aware_shared_candidate.py`, with the stage recorded
per trace and covered by a regression test.

The strict CUDA panel is
`results/negative_cause_diagnosis_20261015/staged_transfer_cuda_panel_32x16_seed20261029/`.
With the same boundary/root gate, `32/16` split, action bank and phase-aware
protocol, the staged arm obtains `5/16` strict pose success, final XY
`36.26 px`, final angle `0.445 rad`, and boundary contact loss `0`.  This is a
point-estimate improvement over the same-run global arm (`4/16`), but the
bootstrap intervals overlap and the XY tube is not reliably reached.  It is
not a stable adapter result.

To test whether the remaining failure was simply a missing task-directed
translation candidate, the same protocol was rerun with fixed goal-directed
candidate magnitudes `(0.10, 0.16, 0.24)`:
`results/negative_cause_diagnosis_20261015/staged_goal_action_cuda_panel_32x16_seed20261030/`.
The staged arm fell to `2/16`, with final XY `31.61 px`, final angle `0.517 rad`,
and boundary contact loss `0`.  Adding goal-directed candidates therefore did
not repair the closed loop and is not promoted.

This agrees with the independent held-out coverage artifacts: one-step pose
coverage is about `0.50`, while unrestricted two-action coverage is about
`0.56`; candidate availability is a real necessary condition, but increasing
the bank alone does not solve learned ranking or multi-step transfer.  The
remaining research object is therefore a transfer-sufficient multi-step
action interface, not another global angle weight or larger primitive fan.

## Fixed two-pulse macro and failure localization

To test whether the staged primitive controller fails because translation and
rotation compete within one pulse, the runner now includes a fixed
translation-pulse -> fresh-observation -> orientation-pulse macro. Both
pulses are selected from the learned action-effect interface; runtime does not
read a simulator successor. With `pulse_steps=2` and `macro_pulse_steps=1`,
the macro consumes the same 24 environment steps as the primitive
`closure_steps=12` protocol.

The full CUDA `32/16` panel is
`results/negative_cause_diagnosis_20261015/two_pulse_macro_reobserve_cuda_panel_32x16_seed20261029/`.
Relative to the same-run staged primitive arm, the staged macro reduces
predicted/observed effect error (`0.2503` to `0.1611`) and final XY
(`36.26` to `29.69 px`), but strict pose success is only `4/16` versus
`5/16`; it is a mechanism improvement, not closure.

The follow-up adds a frozen local moment-sign gate for the second pulse. The
artifact is
`results/negative_cause_diagnosis_20261015/staged_feedback_macro_cuda_panel_32x16_seed20261029/`.
It does not improve strict pose (`4/16`) and slightly worsens final angle
(`0.431` versus `0.416 rad`). A second follow-up keeps the existing `28 px`
XY gate inside the macro, continuing translation until the observed state is
inside the tube before enabling orientation. Its artifact is
`results/negative_cause_diagnosis_20261015/staged_phase_gate_macro_cuda_panel_32x16_seed20261029/`.
It reaches `4/16` strict pose, final XY `28.25 px`, final angle `0.406 rad`,
and predicted/observed effect error `0.1525`. The positive effect-transfer
and XY shifts do not translate to a strict-pose gain.

Together with the prior sign-feedback and candidate-augmentation panels, this
rules out three narrow repairs: insufficient re-observation frequency alone,
wrong first-pulse moment sign alone, and premature phase switching alone. The
remaining boundary is a multi-step within-contact action interface whose
translation and orientation effects are not jointly identifiable from the
current learned model and action support. The macro arms remain diagnostic
controls; they do not establish stable adapter closure or frozen visual
planning.

## Joint two-pulse ranker and protocol-mismatch repair

The next intervention fits a training-only joint two-pulse final-pose ranker
from training-root pairs. At deployment it uses the learned one-step model to
imagine the first pulse, ranks the second candidates jointly, executes the
first pulse, re-observes, and ranks the second pulse again. The initial
implementation exposed a protocol mismatch: the ranker was trained with
`pulse_steps=2` while deployment used `macro_pulse_steps=1`. The runner now
records and enforces the same macro pulse length for training and deployment.

The matched CUDA `32/16` artifact is
`results/negative_cause_diagnosis_20261015/joint_macro_ranker_matched_cuda_panel_32x16_seed20261029/`.
The joint arm reaches `5/16` strict pose success, final XY `30.16 px`, final
angle `0.528 rad`, physical boundary contact loss `0`, and
predicted/observed effect error `0.157`. Thus the mismatch was a genuine
implementation cause of the earlier joint-ranker failure, but correcting it
does not produce an independent downstream gain over the staged arm. Joint
multi-step fitting improves local effect transfer without resolving the
translation/orientation identifiability boundary.

## XY-shield intervention and pooled failure boundary

The next intervention targets the remaining failure mechanism directly.  The
new `xy_shield_mask` is a fixed, successor-free gate for the orientation half
of the two-pulse macro: an orientation candidate is admissible only when the
learned one-step successor keeps predicted XY error within `2 px` of the
current observed XY error.  If no candidate satisfies the tube, the controller
fails closed to the candidate with the smallest predicted XY regression.  The
gate uses no held-out outcome or simulator successor and is covered by the
PushT regression suite.  The implementation and protocol are in
`le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py`.

The three-seed CUDA root-disjoint panel is
`results/negative_cause_diagnosis_20261015/xy_shield_panel_3seed_20261015/`.
It uses 128 training roots, 16 held-out roots per seed, the same boundary
contact protocol, action bank, pulse budget, and `64`-tree models across all
seeds.  Pooled root-level results are:

| controller | strict pose | final XY (px) | final angle (rad) | effect error | boundary contact loss |
|---|---:|---:|---:|---:|---:|
| global one-step | 15/48 (31.25%) | 37.29 | 0.347 | 0.144 | 0/48 |
| staged phase macro | 12/48 (25.00%) | 29.42 | 0.386 | 0.246 | 0/48 |
| XY-shield phase macro | 15/48 (31.25%) | 18.44 | 0.499 | 0.219 | 0/48 |
| joint two-pulse macro | 15/48 (31.25%) | 49.05 | 0.434 | 0.291 | 0/48 |

Against the paired global baseline, the XY shield changes final XY by
`-18.85 px` (95% CI `[-27.88, -10.09]`) and final angle by `+0.152 rad`
(95% CI `[+0.100, +0.203]`).  Its strict-pose delta is exactly `0.0`
(95% CI `[-0.104, +0.104]`).  Thus this intervention confirms the causal
diagnosis—orientation pulses were reliably damaging translation—but does not
solve pose closure.  The positive result is a stable translation-preservation
mechanism, while the remaining failure is within-contact angle
identifiability/action support.  It is not evidence for a stable learned
adapter, frozen RGB online planning, causal latent physics, or universal
world-model transfer.

## Signed orientation-effect follow-up

To separate goal residual from the learned action effect, the runner now fits a
training-root-only signed delta ranker for
`wrap(angle_after - angle_before)`. Its training pulse length is explicitly
matched to `macro_pulse_steps`; deployment remains orientation-first, followed
by a fresh observation and a translation pulse. The implementation also
contains two predeclared controls: the same ranker with the fixed `2 px` XY
shield, and a fixed equal-weight dual ranker averaging the absolute post-pulse
angle-error and signed-delta predictions. No held-out outcome determines the
weight or shield threshold.

CUDA smoke artifacts (8 training roots, 8 root-disjoint held-out roots, seed
`20261030`) are:

- `results/negative_cause_diagnosis_20261015/orientation_delta_ranker_smoke_cuda_20261015/`;
- `results/negative_cause_diagnosis_20261015/orientation_delta_ranker_shielded_smoke_cuda_20261015/`;
- `results/negative_cause_diagnosis_20261015/orientation_dual_ranker_smoke_cuda_20261015/`.

The signed ranker lowers final angle error relative to the absolute ranker
(`0.549` vs `0.656 rad`) but increases final XY (`37.97` vs `23.68 px`) and
reaches only `2/8` strict pose successes. The shielded signed arm changes
these to `0.581 rad`, `30.69 px`, and `2/8`; the dual arm gives `0.651 rad`,
`23.62 px`, and `3/8`, essentially reproducing the absolute arm. Physical
boundary contact loss is `0/8` for all three. Thus the new method confirms a
useful local effect: signed deltas improve angular prediction, but the gain
does not survive the translation-orientation planning interface. Because the
smoke has no independent downstream gain, it is not promoted to a formal
three-seed claim. The current negative-result diagnosis is therefore
sharpened to joint within-contact action identifiability, rather than a mere
absolute-vs-delta target mismatch.

## Non-PushT continuation: PointMaze CUDA temporal/action interventions

To avoid treating PushT as a universal environment, the same frozen-model
diagnostic was run on the locally available JEPA-WM PointMaze holdout. The
protocol uses episodes 1800--1899, 100 root-disjoint holdout episodes, two
causal windows per episode, context length 2, horizons 1/2/4, and the fixed
epoch-50 PointMaze checkpoint. All three runs resolved to the host CUDA
device (`NVIDIA GeForce RTX 5060 Ti`, PyTorch `2.12.0+cu130`):

- `results/pointmaze_cuda_temporal_followup_20261015/observed.json`;
- `results/pointmaze_cuda_temporal_followup_20261015/zero.json`;
- `results/pointmaze_cuda_temporal_followup_20261015/reverse.json`.

The observed-action run has latent prediction error `4.974/4.900/4.711` at
horizons 1/2/4 and covariance-predictor commutator `29.38/32.84/59.42`.
Replacing the action blocks with zero actions changes these to
`4.999/4.856/4.620` and `18.15/21.63/45.39`; reversing the action blocks
changes them to `4.996/4.902/4.732` and `37.60/50.37/70.65`. The action
intervention therefore changes the learned operator geometry in PointMaze,
especially at longer horizons, but it does not by itself establish planning
benefit or a universal commutator certificate. This is independent
non-contact environment evidence for action-conditioned geometry, not a
replacement for a held-out closed-loop planning gate.

TwoRoom dataset replay acquisition was attempted through the official mirror,
but the large archive stalled after a resumable 20 MB partial file; the
process was stopped and the partial cache was retained. The live simulator
results below are independent of that incomplete dataset replay and do not
claim dataset-replay evidence.

## PointMaze support-matched protocol repair and full non-PushT panel

The first support-matched PointMaze external evaluator exposed two independent
protocol defects. The multi-window enriched-export writer reused a stale
batch-local `source_index`, duplicating later-horizon state labels across
windows. The support-matched simulator also advanced one MuJoCo step per
recorded action, although the released PointMaze environment uses
`frame_skip=5`. The latter changed the offline intervention labels and made
the earlier support-matched action-risk result incomparable to the learned
five-step block clock.

Both defects are now repaired without overwriting the original artifacts:

- `experiments/theory_spectral_transfer/run_jepa_wm_pointmaze_temporal_eval.py`
  computes the source index per record and horizon iteration;
- `experiments/empirical_spectral_audit/repair_pointmaze_export_replay_labels.py`
  writes a separate repaired external export;
- `experiments/empirical_spectral_audit/evaluate_pointmaze_support_matched_external_evidence.py`
  uses five physics steps per action and records this in protocol metadata;
- `tests/test_pointmaze_support_matched_protocol.py` checks evaluator/reference
  replay equivalence;
- `experiments/empirical_spectral_audit/run_pointmaze_support_matched_panel.py`
  reruns the frozen panel on CUDA.

The 50-checkpoint corrected panel (two architectures, five seeds, five stages,
20 root-disjoint external roots per checkpoint) is at
`le-wm-main/results/pointmaze_support_matched_external_panel_cuda_20261015_corrected5step/`.
All 50 runs completed on the RTX 5060 Ti with zero failures and zero replay
equivalence error. Mean action-inverse-risk / identifiability / planning-risk
by architecture and stage are:

| architecture | stage 5 | stage 10 | stage 25 | stage 50 | stage 75 |
|---|---:|---:|---:|---:|---:|
| DINO risk | 1.0345 | 1.1290 | 1.3811 | 1.3130 | 1.3053 |
| DINO identifiability | 0.4767 | 0.4967 | 0.5600 | 0.5467 | 0.5400 |
| DINO planning risk | 0.5787 | 0.4309 | 0.3689 | 0.3893 | 0.3979 |
| JEPA risk | 0.9836 | 1.0277 | 1.1645 | 1.1463 | 1.1523 |
| JEPA identifiability | 0.5833 | 0.5800 | 0.5767 | 0.5700 | 0.5700 |
| JEPA planning risk | 0.8166 | 0.7390 | 0.6794 | 0.6794 | 0.6794 |

The corrected protocol removes the replay artifact, but it does not yield a
stable positive learned action-transfer result: identifiability remains
moderate and no cross-architecture planning gate is passed. Therefore the
previous support-matched PointMaze numbers are superseded for scientific
interpretation; the corrected panel is the admissible evidence. This is a
non-PushT protocol repair and failure-localization result, not evidence for a
universal world model, causal latent physics, or a commutator certificate.

## Expanded contact action-support smoke: fail-closed decision

The expanded normal-magnitude bank was rerun on CUDA with 8 training and 8
root-disjoint held-out roots. The final artifact is
`results/negative_cause_diagnosis_20261015/expanded_normal_bank_smoke_cuda_20261015/`.
Adding normal magnitudes `0.04/0.08` while retaining tangent magnitudes
`0.06/0.10/0.16` reduces final XY from `45.38` to `14.48 px`, but changes
strict pose from `1/8` to `0/8` and final angle from `1.629` to `1.781 rad`;
physical boundary contact loss remains `0/8`. Thus the apparent translation
gain does not satisfy the predeclared downstream criterion. This branch is
closed as an action-support-only repair and is not promoted to a formal
multi-seed intervention. The current PushT failure boundary remains joint
within-contact translation/orientation identifiability.

## Live frozen-checkpoint transfer beyond PushT

The non-PushT audit was extended to live rendered environments using frozen
LeWM checkpoints, disjoint action-statistics calibration episodes, and no
adapter training. TwoRoom completed with its environment checkpoint at
`results/live_cross_environment_prediction_20261015/tworoom_cuda/`; Reacher
completed with its checkpoint at
`results/live_cross_environment_prediction_20261015/reacher_cuda/`.
Both use CUDA inference, 24 test episodes, two windows per episode, and
horizons 1/2/4. In-domain action interventions are positive: for TwoRoom,
zero-action minus observed MSE is `+0.285/+0.469/+0.466` at h1/h2/h4, with
95% bootstrap intervals excluding zero; for Reacher it is
`+0.494/+0.903/+1.155`, also excluding zero. This confirms that the frozen
predictors use action information outside PushT, but remains an
action-conditioned prediction result rather than a planning result.

The stricter same-checkpoint cross-environment controls are:

- `results/live_cross_environment_prediction_20261015/tworoom_checkpoint_cross_env_cuda/`;
- `results/live_cross_environment_prediction_20261015/reacher_checkpoint_cross_env_cuda/`.

They expose the transfer boundary. The TwoRoom checkpoint on Reacher has
observed h1/h4 MSE `6.97/8.22`, and zero-action changes MSE by
`-0.245/-0.517`; the Reacher checkpoint on TwoRoom has observed h1/h4 MSE
`0.46/1.28`, and zero-action changes MSE by `-0.230/-0.615`. Thus the same
action-conditioned signal that is predictive in-domain is not preserved by
naive cross-environment checkpoint transfer. This is evidence for an
environment-specific observation/action geometry interface, and directly
rules out claiming a universal world model from the in-domain positives.

Reacher initially failed because the installed dm-control/MuJoCo APIs were
incompatible and headless rendering lacked an explicit backend. The project
now records the CUDA device, `MUJOCO_GL=egl`, and an explicit missing-field
compatibility shim in the artifact protocol. The shim skips only absent
named-index fields and does not synthesize runtime state; its use is an
environment repair, not a learned-model repair.

### Calibration-only affine interface test and OGB-Cube control

To test whether the cross-environment failure was only a latent-coordinate
offset, the audit was extended with a diagonal affine prediction-to-target
map fit on calibration episodes and evaluated on disjoint test episodes. The
artifacts are
`results/live_cross_environment_prediction_20261015/{tworoom_checkpoint_affine_cross_env_cuda,reacher_checkpoint_affine_cross_env_cuda}/`.
The map lowers some absolute errors, but does not restore action
identifiability: TwoRoom-checkpoint -> Reacher changes the raw
zero-minus-observed h1/h4 effect from `-0.245/-0.517` to
`-0.028/+0.024`, while Reacher-checkpoint -> TwoRoom changes
`-0.230/-0.615` to approximately zero at all horizons. This is a
calibration collapse, not a successful transfer repair; only 16 calibration
windows support a 192-dimensional latent, so this map remains diagnostic.

As a third non-PushT in-domain control, the OGB-Cube checkpoint was evaluated
on `swm/OGBCube-v0` with the same CUDA, split, and horizon protocol at
`results/live_cross_environment_prediction_20261015/cube_cuda/`. Its
zero-minus-observed effects are `-0.009/+0.016/+0.023` at h1/h2/h4, with
bootstrap intervals crossing zero. Current evidence therefore supports
environment-conditional action geometry, not a stable all-environment claim;
the next repair target is action support and action semantics rather than
another latent-coordinate postprocess.

### Calibration-only action-interface adapter

The next intervention fits a low-dimensional linear map from target-domain
actions to the frozen checkpoint's action coordinates on calibration episodes
only, with identity regularization and a disjoint test evaluation. The code
path is in
`experiments/audit_frozen_cross_environment_prediction.py`; artifacts are
`results/live_cross_environment_prediction_20261015/{tworoom_checkpoint_action_adapter_cross_env_cuda,reacher_checkpoint_action_adapter_cross_env_cuda}/`.
For TwoRoom-checkpoint -> Reacher, raw h1/h2/h4 effects
`-0.341/-0.514/-0.422` move to `-0.025/-0.066/-0.101`; for
Reacher-checkpoint -> TwoRoom, raw `-0.223/-0.329/-0.476` moves to
`-0.0003/-0.0007/-0.0019`. The source-domain controls remain positive, but
neither target-domain direction becomes reliably positive. The adapter fixes
part of the action-coordinate mismatch and therefore improves the diagnosis,
but the residual boundary is a deeper environment-specific visual/dynamics
interface mismatch. It is not yet a cross-environment planning solution.

The same adapter has a useful in-domain control on OGB-Cube at
`results/live_cross_environment_prediction_20261015/cube_action_adapter_cuda/`:
the raw zero-minus-observed effects are `+0.018/+0.050/+0.039` at h1/h2/h4,
whereas the adapted effects are `+0.039/+0.082/+0.161`; all adapted bootstrap
intervals exclude zero and all 16 paired test episodes have zero action worse
than observed action. This is the first non-PushT positive result for the new
repair, but it is still a single-checkpoint prediction result and requires
fresh seeds and an independent planning consequence before promotion.

Two additional Cube seeds were then run at
`results/live_cross_environment_prediction_20261015/{cube_action_adapter_seed2_cuda,cube_action_adapter_seed3_cuda}/`.
Across all three seeds, the adapted h1/h2 effects are positive with
episode-bootstrap intervals excluding zero: respectively
`(+0.039,+0.082)`, `(+0.036,+0.050)`, and `(+0.022,+0.080)`. At h4 the
point estimates remain positive (`+0.161/+0.088/+0.082`), but the third
seed interval crosses zero. The supported result is therefore a three-seed
h1/h2 prediction gain and a positive-but-not-yet-conclusive h4 trend, not a
stable closed-loop or universal claim.

### Independent live RGB consequences outside PushT

The first live Cube planning smoke exposed a goal-interface mismatch: the
installed CubeEnv keeps the rendered target on `_cur_goal_rendered` and returns
the state target as `target`, while `WorldModelPolicy` requires `goal` pixels.
`experiments/run_live_cube_action_adapter_planning.py` now uses a raw-env
goal alias, explicitly starts Cube in `mode="task"`, requests
`render_goal=True`, and fails closed if no rendered target is available. The
artifact `results/live_cube_action_adapter_planning_20261015/smoke_2ep/`
confirms that the live RGB CEM loop reaches the policy and simulator.

The corrected frozen protocol uses host CUDA, EGL rendering, identical CEM
parameters, eight fixed initial seeds, `reset_mode="wait"`, and seed-sorted
pairing. The full Cube artifact is
`results/live_cube_action_adapter_planning_20261015/paired_8ep_canonical/`:
baseline and calibration-only action-adapter planning are both `0/8`, with
`0` wins, `0` losses, and `8` ties. Thus the three-seed h1/h2 prediction gain
does not produce an independent Cube planning consequence under this frozen
visual CEM protocol.

To add a non-contact environment rather than relying on PushT, a live RGB
TwoRoom runner was added at
`experiments/run_live_tworoom_planning.py`. It aliases only TwoRoom's
rendered target image, keeps simulator state out of the policy, and compares
the frozen TwoRoom LeWM checkpoint with a same-seed random controller. The
strict paired artifact is
`results/live_nonpusht_planning_20261015/tworoom_paired_8ep_canonical/`.
Frozen LeWM obtains `2/8`, random obtains `1/8`; seed-level pairing gives one
win, zero losses, seven ties, and a simple episode bootstrap of
`[0.000, 0.375]`. This is a promising non-PushT point estimate, not stable
evidence of a general planning gain.

An earlier TwoRoom run used completion-order arrays and is retained only as a
debug artifact; the canonical run sorts outcomes by seed and records the
paired delta explicitly. The current cross-environment conclusion is therefore
more precise: in-domain action-conditioned prediction is positive in TwoRoom,
Reacher, and selected Cube horizons; only TwoRoom has a preliminary live
planning consequence; Cube's adapter does not yet transfer its prediction
improvement to planning; and no universal world-model or commutator claim is
opened.

### Multi-panel TwoRoom consequence replication

The preliminary TwoRoom result was expanded without changing the checkpoint,
goal alias, CEM configuration, action normalization, or runtime input. Three
non-overlapping 8-episode panels were run on host CUDA with seeds
`20261015`, `20261023`, and `20261031`; each panel contains a same-seed random
control and frozen LeWM arm. The panel artifacts are
`results/live_nonpusht_planning_20261015/tworoom_panel_seed20261015/`,
`results/live_nonpusht_planning_20261015/tworoom_panel_seed20261023/`, and
`results/live_nonpusht_planning_20261015/tworoom_panel_seed20261031/`.

The predeclared aggregate is
`results/live_nonpusht_planning_20261015/tworoom_aggregate_24ep/planning_aggregate.json`.
Random succeeds on `2/24` episodes (`8.33%`), frozen LeWM on `7/24`
(`29.17%`). Seed-level paired outcomes are `5` wins, `0` losses, and `19`
ties, giving `+0.2083` success-rate difference with bootstrap 95% interval
`[+0.0417,+0.3750]`; the one-sided exact sign-test p-value is `0.03125`.

This is now an independent non-PushT planning consequence with fixed protocol
and cross-panel replication. Its claim remains scoped to the TwoRoom task and
checkpoint: it does not establish cross-environment transfer, a universal
world model, stable pose closure, or a commutator certificate. The unchanged
Cube result (`0/8` for both baseline and action adapter) remains an explicit
boundary showing that a prediction improvement need not become a planning
improvement.

### Multi-panel Reacher consequence replication

The non-contact Reacher branch was then implemented and run independently of
PushT. `experiments/run_live_reacher_planning.py` uses the frozen Reacher
checkpoint and the same CEM/action-normalization protocol as the other live
runners. Because the installed qpos-match environment has no native rendered
goal field, `ReacherGoalImageWrapper` renders a fixed target-qpos offset after
reset, restores the live state, and exposes only the rendered goal image to the
policy. The wrapper removes qpos, qvel, target position, and finger position
from the policy info dict. The runner records `device=cuda`, EGL rendering,
frozen weights, and same-seed pairing in each artifact.

Three non-overlapping 8-episode panels used seeds `20261015`, `20261023`, and
`20261031`, with the released `qpos_match` checkpoint, horizon `5`,
receding horizon `5`, action block `5`, and CEM `64/8/8`. Artifacts are
`results/live_nonpusht_planning_20261015/reacher_panel_seed20261015/`,
`results/live_nonpusht_planning_20261015/reacher_panel_seed20261023/`, and
`results/live_nonpusht_planning_20261015/reacher_panel_seed20261031/`.
The preregistered aggregate is
`results/live_nonpusht_planning_20261015/reacher_aggregate_24ep/planning_aggregate.json`.

Random succeeds on `0/24`; frozen LeWM succeeds on `16/24` (`66.67%`). The
paired episode outcomes are `16` wins, `0` losses, and `8` ties, giving a
success-rate difference of `+0.6667`, bootstrap 95% interval
`[+0.4583,+0.8333]`, and one-sided exact sign-test `p=1.53e-05`. This is a
replicated non-PushT live RGB planning consequence for Reacher. It does not
remove the Cube boundary, does not validate the missing Reacher dataset replay
protocol, and does not establish a universal world model or commutator
certificate. The official Reacher HDF5 download remains environment-blocked by
the host's `Network is unreachable` response from HuggingFace.

### Reacher target-geometry robustness boundary

To test whether the Reacher consequence was tied to one goal construction, the
runner was extended with a predeclared `--target-delta` and no other protocol
change. Three fresh 8-episode panels used target-qpos offsets
`[+0.65,-0.65]`, `[+0.65,+0.65]`, and `[-0.65,+0.65]`; all used CUDA, the same
frozen checkpoint, CEM `64/8/8`, horizon `5`, receding horizon `5`, action
block `5`, and same-seed random controls. The complete condition aggregate is
`results/live_nonpusht_planning_20261015/reacher_target_geometry_aggregate/target_geometry_aggregate.json`.

The pooled result is Random `1/24`, frozen LeWM `6/24`, with paired `6` wins,
`1` loss, and `17` ties; success-rate delta is `+0.2083`, bootstrap 95%
interval `[0.0000,+0.4167]`, and exact sign-test `p=0.0625`. Per-condition
success deltas are `+0.25`, `+0.25`, and `+0.125` respectively, but every
condition-level interval crosses zero. This supports directional robustness
as a point estimate, not a statistically closed universal planning claim.
The stronger earlier Reacher panel (`16/24`) therefore remains a
checkpoint/seed-panel result under the original goal geometry; this sweep
identifies target geometry and action support as remaining variance sources.

### Cube official receding-horizon protocol closure

Because the initial Cube planning runner used `receding_horizon=1`, we ran a
fresh protocol-matched control using the released Cube evaluation settings:
frozen `swm/OGBCube-v0` checkpoint, live RGB observations, CUDA/EGL, horizon
`5`, receding horizon `5`, action block `5`, CEM `64/8/8`, `var_scale=1.0`,
75-step episode budget, calibration-only action adapter, and the same eight
seeds for baseline and adapter. A two-episode smoke artifact is retained at
`results/live_cube_action_adapter_planning_20261015/receding5_smoke_2ep/`;
the fresh eight-episode panel is
`results/live_cube_action_adapter_planning_20261015/receding5_official_8ep/`.

Both arms obtain `0/8` successes, with `0/0/8` paired wins/losses/ties. The
evaluator-only final block-goal distance is `0.1460 m` for baseline and
`0.1500 m` for the adapter; the adapter is closer on `3/8` episodes and
farther on `5/8`. Together with the earlier `var_scale=0.35/0.50` and
150-step traces, this rules out receding-horizon mismatch, initial CEM
variance alone, and the short episode budget as sufficient explanations for
the Cube negative planning result. The remaining supported diagnosis is a
goal-conditioned action-to-task transfer failure: the calibration adapter
improves selected h1/h2 prediction effects but does not preserve the
goal-directed action ranking in the frozen visual CEM loop. This is a
non-PushT failure boundary, not evidence for a universal planning claim.

### Cube action-space repair and strict pairing audit

The Cube protocol was then audited at the environment interface. OGBench's
`ManipSpaceEnv` expects normalized actions in `[-1,1]`, while the existing
z-score inverse transform could emit unbounded values from CEM candidates. The
runner now exposes this as an explicit `--clip-env-actions` intervention and
records pre-clip fraction and pre/post-clip maxima; prior artifacts are left
unchanged. In the CUDA/EGL smoke, 13.7% of baseline and 11.7% of adapter
inverse-transformed values were outside the environment bounds, with maxima of
5.20 and 4.08. This confirms a real protocol defect, but does not by itself
explain the planning failure.

The audit also found that Cube's visual variation reset uses an unseeded
variation space, so equal integer seeds did not guarantee equal initial
states. The live runner was repaired to snapshot baseline `qpos/qvel` and
variation values and restore them for the adapter arm. The strict matched
CUDA/EGL panel is
`results/live_cube_action_adapter_planning_20261015/bounded_matched_8ep_v2/`;
both arms use the official horizon `5`, receding horizon `5`, action block `5`,
CEM `64/8/8`, `var_scale=1.0`, 75 steps, and bounded environment actions.
Initial conditions are restored exactly. Both arms remain `0/8` success and
have identical final distance on all eight pairs; the paired distance delta is
`0.0` throughout. The action output still requires clipping on 13.43% of
baseline and 13.03% of adapter values. Thus action clipping fixes a genuine
interface violation but yields no independent Cube planning gain.

The corresponding strict ranking audit is
`results/live_cube_action_ranking_20261015/bounded_matched_audit25_8ep/`.
After 25 steps, neither arm changes block-goal distance beyond approximately
`5.2e-9 m`; all eight paired differences are zero. This identifies the
remaining near-term mechanism boundary as lack of contact/approach-phase
coverage and goal-conditioned action-effect transfer, rather than only an
unbounded-action bug. Cube therefore remains a non-PushT negative planning
control, while TwoRoom and Reacher remain separately scoped positive
consequences.

### Cube action-support intervention: approach phase is a real boundary

To distinguish a latent/action-ranking failure from an action-support failure,
`experiments/audit_cube_action_interventions.py` runs strict paired raw-physics
interventions on all five predefined Cube tasks and two seeds per task. Each
condition restores the same `qpos/qvel` and variation values; privileged cube
and end-effector measurements are read only after each action block. The
artifact is `results/cube_action_interventions_20261015_full.json`.

The environment interface is explicit: the normalized five-dimensional action
is relative end-effector x/y/z, yaw, and gripper control; one control step
expands to 25 physics steps. A vertical descent alone does not test contact
when the initial cube is laterally offset. The diagnostic bank therefore
includes zero/single-axis controls, a 15-step descent, and a separate
`align_descend_close_push` sequence. The latter uses the measured initial
cube-to-effector direction only as a mechanism probe, not as a policy input or
planning result.

The earlier no-contact result was reproduced: zero and single-axis controls
move the cube only `3.93e-5 m` from gravity settling, and descent without
lateral alignment can remain ineffective. After adding horizontal alignment,
the action-support probe produces nontrivial cube motion in `3/10` paired
task-seed cases (maximum displacement about `0.073 m`) and reduces goal
distance in those cases. The unaligned
`approach_close_push_goal_direction` condition produces cube motion in `2/10`
cases, including two cases with positive reported gripper-contact steps. The
remaining task/seed heterogeneity is important: this is not yet a robust Cube
control method, but it proves that the simulator/action interface can produce
task-relevant effects when the approach phase is covered.

This changes the diagnosis of the earlier `0/8` live Cube planning result. It
cannot be attributed to an incapable environment or only to latent prediction;
the frozen CEM protocol does not reliably discover the required lateral
alignment, descent, and push phase from RGB goal-conditioned inputs. The next
Cube intervention is therefore a predeclared action-support candidate bank or
phase-conditioned planner, followed by a fresh paired CUDA/EGL planning test.
Until that test is run, the result remains an action-support/goal-conditioned
transfer boundary, not a repaired planning claim.

### Phase-conditioned candidate-bank intervention

The next intervention implemented the candidate-support repair directly in
`experiments/run_live_cube_phase_conditioned_planning.py`. It keeps the Cube
checkpoint frozen and predeclares four generic x+/x-/y+/y- action phases. Each
phase contains lateral alignment, two descent blocks, gripper closure, and a
short push. The frozen LeWM cost ranks these candidates, and ordinary CEM is
initialized from the selected candidate. No cube position, simulator state, or
planning outcome is used by the bank. The solver also preserves the selected
remaining phase across short-pulse visual re-observations.

The strict CUDA/EGL 8-episode artifact is
`results/live_cube_phase_conditioned_planning_20261015/paired_8ep/planning.json`.
It uses the same frozen checkpoint, action clipping, horizon `5`, action block
`5`, CEM `64/8/8`, 75-step budget, and exact initial-state pairing as the
official Cube control. The phase bank was selected 24 times, but baseline and
phase-conditioned planning both remain `0/8`; paired success delta is `0`, and
the mean final-distance delta is `+0.00114 m` (phase worse). A two-episode
short-pulse smoke with `receding_horizon=1` also gives identical paired
outcomes.

Thus candidate support and re-observation alone do not solve Cube. The bank is
an executed negative intervention, not a positive method claim: the remaining
failure is specifically that frozen RGB goal-conditioned latent costs do not
rank contact-phase actions in a decision-useful way. This strengthens the
action-identifiability boundary and prevents attributing the Cube failure only
to CEM variance or missing approach candidates.

## Non-PushT cross-environment continuation

To prevent PushT from carrying the entire planning argument, the fixed live
RGB protocol was extended with a fourth TwoRoom panel and a fourth Reacher
panel. Both were run with the frozen checkpoint, identical CEM/action/history
settings, paired random controls, and host CUDA. The Reacher run additionally
uses `MUJOCO_GL=egl`; the earlier GLFW failure was an environment-rendering
issue, not a CPU fallback or a model change.

The new panel artifacts are:

- `results/live_nonpusht_planning_20261015/tworoom_panel_seed20261039/`;
- `results/live_nonpusht_planning_20261015/reacher_panel_seed20261039/`.

The four-panel aggregates are:

- `results/live_nonpusht_planning_20261015/tworoom_aggregate_32ep/planning_aggregate.json`;
- `results/live_nonpusht_planning_20261015/reacher_aggregate_32ep/planning_aggregate.json`.

TwoRoom obtains `6/32` paired wins, no losses, and a success-rate delta of
`+0.1875` with bootstrap CI `[+0.0625,+0.34375]` (exact one-sided sign-test
`p=0.015625`). Reacher obtains `21/32` wins, no losses, and a delta of
`+0.65625` with CI `[+0.5000,+0.8125]` (`p=4.77e-7`). These are environment-
scoped live planning consequences, not a universal transfer result.

The cross-environment evidence index is generated by
`experiments/summarize_cross_environment_planning.py` and stored at
`results/cross_environment_planning_20261015/nonpusht_evidence.json` (and its
Markdown companion). It deliberately does not pool incompatible estimands:
TwoRoom/Reacher are live RGB paired planning, Cube is a strict live negative
boundary (`0/8` versus `0/8`), and PointMaze is an independent held-out offline
gate with mean gain `+0.1007` but CI `[-0.0277,+0.2478]` and `p=0.0877`.
Therefore the current non-PushT evidence supports scoped positive consequences
in TwoRoom and Reacher, while Cube and PointMaze preserve important failure and
uncertainty boundaries. It does not support a universal world model or a
universal commutator certificate.

### Prediction-to-planning propagation audit

The companion artifact
`results/cross_environment_planning_20261015/prediction_to_planning_gap.json`
joins the existing CUDA frozen-prediction audits with the paired planning
aggregates. It is descriptive rather than a fitted cross-environment
correlation, because the latent MSE scales are environment-specific. The
mechanistic split is clear: TwoRoom and Reacher have positive observed-vs-zero
action-effect contrasts at `h=1,2,4` and positive live planning consequences;
Cube has measurable prediction MSE (`h=4` mean `0.4414`) but its action-effect
contrast CI crosses zero and its strict live planning consequence remains
`0/8`; PointMaze has no matched prediction join and its independent planning
gate remains statistically inconclusive. This identifies action
identifiability—not prediction MSE alone—as the next intervention target for
Cube and as a necessary condition before promoting any metric to a planning
certificate.

### Full TwoRoom latent/planning alignment audit

The full CUDA TwoRoom audit was completed from the frozen `tworoom/lewm`
checkpoint using the downloaded `tworoom.h5` dataset. Phase 1 processed
`1,592,619` samples and produced
`results/tworoom_full_spectral_cuda_20261015/full_latents/latents.npz` with
200,000 rich rows retained for downstream audits. Phase 2 processed
`3,716,111` endpoint rows and completed at
`results/tworoom_full_spectral_cuda_20261015/full_latents/planning_alignment.json`.

The vanilla latent endpoint cost has Spearman correlation `0.6653` with
physical state distance, but only `0.2034` with goal distance and `-0.6621`
with the success proxy. The best reported spectral projection (top-32) gives
state-distance `0.6560`, goal-distance `0.2393`, and success `-0.6490`.
Thus the spectrum slightly changes task geometry but does not establish a
goal-aligned planning metric; the full scan also removes the overly optimistic
interpretation of the earlier 96-point smoke.

The independent empirical spectral audit over the retained 200,000 rich rows
reports dynamics lift `24.33/17.38/10.41` at `k=4/8/16`, planning lift
`15.97/11.12/6.77`, and nuisance lift `9.50/7.31/4.86`. The corresponding
planning purity is `1.85/1.65/1.47`, while predictor spectral mean cosine is
`0.9605`. These are offline representation diagnostics: they show that
action-conditioned dynamics and planning bands are detectable and stronger
than nuisance in this audit, but they do not prove goal-conditioned action
ranking, online success, universal alignment, or a commutator certificate.

The complete audit artifact is
`results/tworoom_full_spectral_cuda_20261015/full_latents/empirical_spectral_audit.json`.
The current intervention target is therefore not “add more spectral bases”;
it is to make goal/action conditioning explicit and test whether that changes
held-out candidate ranking and closed-loop planning under a fresh protocol.

### Goal/action transfer audit after the full TwoRoom scan

To test the proposed repair without using held-out planning outcomes, the new
`experiments/audit_tworoom_goal_action_transfer.py` performs an episode-disjoint
60/20/20 train/calibration/test audit. It joins only adjacent
`episode_id,timestep` rows to obtain the physical next-state delta, fits a
fixed ridge readout on train episodes, and compares state-action,
latent-action, latent-goal-action, and predictor-effect-goal-action features.
The artifact is
`results/tworoom_full_spectral_cuda_20261015/full_latents/goal_action_transfer_audit.json`.

On the held-out test split, latent-action reduces next-state RMSE from `0.0251`
to `0.0211` relative to state-action and reduces endpoint-goal progress RMSE
from `0.0228` to `0.0197`; adding the exported endpoint-goal field changes
these only to `0.0210` and `0.0197`. The predictor-effect-goal-action variant
is worse (`0.0222` and `0.0208`). Action-vs-zero model contrasts are positive
on about `62.3%` of test rows, but this is not a matched candidate-ranking
metric.

This is a useful negative result for the proposed repair: the latent carries
one-step physical effect information, but simply appending an endpoint goal or
using the frozen predictor effect does not add independent transfer information.
The endpoint goal is a window-derived proxy rather than an external task goal,
so the artifact cannot support a goal-conditioned planning claim. The next
experiment must use a shared candidate bank or a fresh live visual protocol;
another linear goal concatenation would not address the identified bottleneck.

### Fresh TwoRoom receding-horizon intervention

The compatible short-feedback protocol was tested with `horizon=5`,
`action_block=5`, and `receding_horizon=1`; the checkpoint action encoder
requires a five-action block, so the attempted `action_block=1` configuration
is now rejected before inference rather than failing inside `Conv1d`. On a
fresh 8-episode CUDA panel (`seed=20261047`), frozen LeWM obtains `3/8`
successes versus random `0/8`, with paired `3/0/5` wins/losses/ties. Together
with the same-protocol panel at `seed=20261015`, the aggregate is `4/16` wins,
no losses, delta `+0.25`, and bootstrap CI `[+0.0625,+0.50]`; the exact
sign-test is `p=0.0625`, so this is not yet a statistically significant
replication.

The matched fresh-seed comparison between `receding_horizon=1` and the
standard `receding_horizon=5` produced identical LeWM success vectors (`3/8`)
and identical random controls. Thus the observed fresh-panel gain cannot be
attributed to shorter feedback alone. The artifacts are
`results/live_nonpusht_planning_20261015/tworoom_receding1_seed20261047/`,
`tworoom_receding5_seed20261047/`, and
`tworoom_receding1_aggregate16/`. This is scoped TwoRoom evidence, not a
general repair of goal-conditioned action ranking.

### Matched TwoRoom candidate-bank intervention

To separate candidate coverage from latent ranking, the live runner now has a
fixed candidate-bank CEM that injects predeclared zero, axis-aligned, and
diagonal macro-actions into the same `64`-candidate budget. It never reads
simulator state or held-out outcomes. Across two matched 8-episode panels
(`seed=20261047` and `seed=20261055`), candidate-bank and standard CEM have
identical outcomes in the first panel; in the second, candidate-bank has `0/8`
versus standard `1/8`. The paired summary is `0/16` wins, `1/16` losses,
`15/16` ties, delta `-0.0625`, bootstrap CI `[-0.1875, 0]`.

The artifact is
`results/live_nonpusht_planning_20261015/tworoom_candidate_bank_matched/pairwise.json`.
This intervention does not support candidate coverage as the primary TwoRoom
bottleneck; adding generic actions does not improve the frozen latent's
goal-conditioned decision. The remaining target is the ranking/utility
interface itself, not a larger undirected candidate bank.

### Covariance-spectral utility intervention

The frozen LeWM cost is terminal full-dimensional Euclidean latent MSE. To
test whether utility geometry itself is the bottleneck, the live runner now
supports a fixed top-32 covariance spectral projection computed from the
completed TwoRoom latent export. The predictor and encoder remain frozen; only
the CEM terminal cost changes.

Across two matched 8-episode panels (`seed=20261047` and `seed=20261055`),
the spectral utility and Euclidean utility have identical LeWM success vectors
and identical random controls. The paired intervention artifact reports
`0/16` wins, `0/16` losses, and `16/16` ties. This rules out a simple
covariance top-band metric as a practical repair for the current TwoRoom
decision interface. The artifact is
`results/live_nonpusht_planning_20261015/tworoom_spectral32_matched/pairwise.json`.
The remaining target is not merely a covariance projection; it requires a
task/action-conditioned utility or a learned decision readout with independent
held-out validation.

### Latent-to-physical utility intervention

The next intervention fitted a linear ridge readout from latent to the two
TwoRoom physical proprio coordinates using train episodes only. The readout
has held-out test RMSE `0.0864` against target standard deviation `1.0194`,
so physical decoding itself is stable across episode-disjoint splits. It was
then used only as a frozen terminal utility for live RGB CEM; the checkpoint
and predictor remained unchanged. Across the two matched 8-episode panels,
the physical-readout utility loses two episodes and wins none relative to
Euclidean utility (`0/16` wins, `2/16` losses, `14/16` ties, delta `-0.125`).

The paired artifact is
`results/live_nonpusht_planning_20261015/tworoom_physical_readout_matched/pairwise.json`;
the readout audit is
`results/tworoom_full_spectral_cuda_20261015/full_latents/physical_readout/readout.json`.
This is a decisive separation between representation decodability and
decision sufficiency: a stable physical readout does not provide a stable
planning utility. The remaining method must model action-conditioned effects,
not just decode state coordinates.

### Action-conditioned physical utility intervention

To make the utility depend on predicted effect rather than only a decoded
state, `experiments/fit_tworoom_action_conditioned_readout.py` fits a frozen
train-episode readout from predicted `h=5` terminal latent plus the first
normalized action block to the physical `t+5` successor. On the held-out
episode split, adding the action block reduces readout RMSE from `0.6700` to
`0.6474` (target standard deviation `0.9657`).

The corresponding live CEM utility uses this readout for predicted terminals
and the independent train-split physical readout for goal images. It is worse
than default Euclidean utility on the matched live panels: `0/16` wins,
`2/16` losses, `14/16` ties, delta `-0.125`. The artifact is
`results/live_nonpusht_planning_20261015/tworoom_action_conditioned_matched/pairwise.json`;
the offline fit is
`results/tworoom_full_spectral_cuda_20261015/full_latents/action_conditioned_readout/readout.json`.

This closes another tempting repair: better action-conditioned successor RMSE
does not automatically produce better candidate selection. The next decisive
measurement is held-out shared-candidate action-ranking regret, with utility
selected on calibration data and planning outcomes kept untouched.

### Root-disjoint shared-candidate ranking and protocol audit

The new shared-candidate audit first exposed an implementation mismatch: the
audit had repeated the current RGB frame and action into a three-step history,
whereas the live `WorldModelPolicy` receives a single current frame/action
(`history_len=1`). The audit now defaults to the live input contract and records
the history length explicitly in its protocol. The earlier three-frame result
is invalid as evidence about the live controller and is retained only as a
debugging artifact.

With the corrected live-history protocol, 32 fresh root-disjoint roots and a
shared 16-candidate constant-action bank give top-1 `34.4%`, top-4 `53.1%`,
mean true rank `7.41/16`, mean Spearman `-0.004`, and mean regret `42.20`.
This does not establish a useful action ranking. A second 32-root audit with
32 shared multi-stage macro-action sequences gives top-1 `18.8%`, top-4
`31.3%`, mean Spearman `0.050`, and mean regret `31.88`; the larger bank
changes the support and regret scale but does not produce reliable ranking.

An exploratory calibration-only ridge utility trained on eight roots improved
the first small held-out panel's regret but lost top-1 accuracy. On an
independent fresh 16-root held-out panel it increased regret from `49.39` to
`56.23` and reduced top-1 from `31.25%` to `12.5%`. It is therefore not
promoted into the controller. The evidence now localizes the remaining
TwoRoom negative to frozen goal-conditioned action-effect utility/identifiability,
not merely candidate-bank coverage, covariance projection, physical decoding,
or the earlier history-shape bug.

The code artifacts are
`experiments/audit_tworoom_shared_candidate_ranking.py`,
`experiments/audit_tworoom_sequence_candidate_ranking.py`, and
`experiments/audit_tworoom_calibrated_action_ranking.py`; all CUDA runs and
their JSON outputs are under
`results/tworoom_full_spectral_cuda_20261015/`. These remain diagnostics with
offline simulator labels, not claims of online planning or a universal
transfer certificate.

### Latent-action ranker intervention

To test whether the unresolved ranking error could be repaired by a
task-conditioned readout, `experiments/audit_tworoom_latent_action_ranker.py`
fits a low-capacity ridge ranker using only calibration roots. Its features
combine the frozen terminal latent, goal latent, predicted-goal difference,
the first raw macro-action, and the frozen model cost. On two fresh 16-root
held-out panels, the ranker reduced mean replay regret and improved top-4
selection, but exact top-1 selection was not stable.

The saved ranker was then connected to the RGB-only live CEM through the new
`latent_action_ranker` cost metric in
`experiments/run_live_tworoom_planning.py`. In an independent 8-episode,
same-seed, same-candidate-bank comparison, it produced `0/8` wins, `1/8`
loss, and `7/8` ties against the frozen Euclidean utility (success delta
`-0.125`). Thus the offline regret improvement does not transfer to closed
loop planning. The ranker artifact remains a diagnostic/readout interface;
it is not promoted to the formal controller. This is a further localization
of the negative result: even a calibration-only task readout can improve
offline action ranking without supplying a reliable online planning
consequence.

## Fresh non-PushT CUDA replication (seed 20261100)

To test whether the earlier non-PushT planning consequences survive a fresh
seed, the unchanged frozen RGB protocols were rerun with host CUDA access on
the RTX 5060 Ti. The two runs use `PYTHONPATH=.`, `MUJOCO_GL=egl`,
`horizon=5`, `action_block=5`, `receding_horizon=5`, the existing frozen
checkpoints, and paired random controls. No test outcome was used to modify
the controller or select a checkpoint.

The new panel artifacts are:

- `results/live_nonpusht_planning_20261015/tworoom_panel_seed20261100/`;
- `results/live_nonpusht_planning_20261015/reacher_panel_seed20261100/`.

The seed-level aggregates are:

- `results/live_nonpusht_planning_20261015/tworoom_aggregate_40ep/planning_aggregate.json`;
- `results/live_nonpusht_planning_20261015/reacher_aggregate_40ep/planning_aggregate.json`.

TwoRoom is `6/40` paired wins, `0` losses, delta `+0.15`, bootstrap CI
`[+0.05,+0.275]`, one-sided exact sign-test `p=0.015625`. Reacher is `24/40`
wins, `0` losses, delta `+0.60`, bootstrap CI `[+0.45,+0.75]`,
`p=5.96e-08`. The fresh TwoRoom panel itself is a tie (`2/8` versus
`2/8`), while the fresh Reacher panel is positive (`4/8` versus `1/8`).

The refreshed non-PushT evidence index is
`results/cross_environment_planning_20261015/nonpusht_evidence_40ep.json`.
It preserves the incompatible-estimand separation: TwoRoom and Reacher have
scoped live RGB consequences, Cube remains a negative boundary, and PointMaze
remains inconclusive. The result strengthens the Reacher consequence and
maintains the TwoRoom aggregate, but it does not open a universal world-model,
cross-environment transfer, causal-latent-physics, or commutator-certificate
claim.

## Reacher target-geometry robustness replication (48 episodes)

The predeclared target-qpos conditions were expanded with three fresh CUDA/EGL
panels, eight paired episodes per condition, using seeds disjoint from the
original robustness panels. The new artifacts are
`results/live_nonpusht_planning_20261015/reacher_target_delta_pp_20261201/`,
`reacher_target_delta_np_20261209/`, and
`reacher_target_delta_pn_20261217/`. The complete six-panel aggregate is
`results/live_nonpusht_planning_20261015/reacher_target_geometry_aggregate_48ep/target_geometry_aggregate.json`.

Per-condition deltas are `+0.25` for `(+0.65,+0.65)`, `+0.125` for
`(-0.65,+0.65)`, and `+0.25` for `(+0.65,-0.65)`; each condition has only
16 episodes and its interval touches or crosses zero. Pooled across the three
predeclared target geometries and 48 paired episodes, frozen LeWM obtains
`11` wins and `1` loss over random, delta `+0.2083`, bootstrap CI
`[+0.0833,+0.3333]`, and one-sided exact sign-test `p=0.00317`.

This strengthens the Reacher result from a single target geometry to a scoped
target-family robustness finding. It still does not establish a universal
world model, cross-environment transfer, causal latent dynamics, or a
commutator certificate; condition-level uncertainty remains material.

## GDD-WM realized-RGB successor panel: 2 architectures × 5 seeds

The existing dataset-matched RGB successor reports were aggregated without
selecting cases by outcome. The panel contains two predictor architectures
(`mlp` and `standardized_mlp`), five seeds per architecture, and eight
root-disjoint outcome roots per case. All ten reports pass renderer
consistency, realized-RGB availability, and PCA-projected successor gates.
The aggregate artifact is
`results/gdd_wm_rgb_successor_panel_20261015/aggregate.json`.

Across the 80 outcome roots, the candidate-level fraction for which the
action-conditioned prediction is closer to the realized RGB successor than the
source-latent baseline is `0.6656`; the two-level case/root bootstrap CI is
`[0.5977,0.7305]`. Relative L2 improvement is `0.1421` with two-level CI
`[0.0849,0.1965]`. All 10 architecture-seed case means are positive, although
individual roots can still be negative.

This closes the realized-RGB successor subgate across a small learned-model
panel. It does not pass GDD-WM D1: the current candidate bank still lacks
repeated independent candidate-set contexts per root, and the historical
teacher score has exact negative-cost leakage. Therefore D2 distillation is
still gated, and this result is not a D-JEPA teacher, causal-physics, or
closed-loop planning claim.

## PointMaze non-PushT temporal/action intervention (40 held-out episodes)

To keep the evidence portfolio independent of PushT, the frozen JEPA-WM
PointMaze checkpoint was evaluated on the tail holdout `1800--1839` with the
same CUDA runtime and three action conditions: observed within-episode
actions, zero actions, and a cross-episode action permutation. The summary
artifact is
`results/nonpusht_pointmaze_temporal_20261015/action_intervention_summary.json`;
the three raw reports are stored next to it. All three reports declare the
same checkpoint hash, split, horizon set `{1,2,4}`, and `device=cuda`.

| condition | h=1 MSE | h=2 MSE | h=4 MSE |
|---|---:|---:|---:|
| observed actions | 4.9631 | 4.9539 | 4.7527 |
| zero actions | 4.9936 | 4.8476 | 4.6059 |
| episode-shuffled actions | 4.9713 | 4.8788 | 4.6755 |

Observed actions improve over zero at h=1 by `0.0305` MSE and improve over
episode-shuffled actions by `0.0082`, but are worse than both controls at h=2
and h=4. The result is therefore a horizon-dependent action-conditioning
diagnostic, not evidence of reliable action identifiability or planning
transfer. The observed commutator values (`3.9418`, `3.6366`, `3.1802` for
h=1/2/4) also do not provide a monotone decision certificate: the control
values change substantially under the same split and do not track the
prediction ordering. PointMaze is retained as a non-PushT boundary case:
temporal/action signal exists, but its multi-step use is not stable.

The follow-up used the remaining disjoint tail holdout `1840--1899` (60
episodes) under the identical protocol. There, observed actions beat both
controls at all horizons, but the effect was small: observed-minus-zero MSE
was `-0.0312/-0.0267/-0.0089` and observed-minus-shuffled MSE was
`-0.0117/-0.0067/-0.0076` for h=1/2/4. The pooled 100-episode aggregate is
`results/nonpusht_pointmaze_temporal_20261015/action_intervention_panel_aggregate_100ep.json`.
It gives observed-minus-zero MSE deltas `-0.0309/+0.0265/+0.0534` and
observed-minus-shuffled deltas `-0.0103/+0.0260/+0.0263` for h=1/2/4.

Thus the split-level heterogeneity is real and the pooled conclusion is
strictly horizon-conditioned: the checkpoint contains a small short-horizon
action signal, but it does not provide stable h=2/4 action-conditioned
prediction on this PointMaze panel. This is a stronger non-PushT failure
boundary for the proposed transfer-sufficiency claim, not a reason to tune a
selector on the test split.

## GDD-WM repeated candidate-set context construction (D1 structure repair)

The D1 failure was decomposed into a missing experimental structure versus an
invalid teacher. `experiments/gdd_wm/build_repeated_candidate_contexts.py`
joins independently generated candidate banks by exact action-sequence hash,
rather than by candidate index, and preserves source bank hash, teacher
provenance, root, split, and runtime references. It never creates teacher or
execution labels.

Using two existing PointMaze candidate-bank artifacts, the builder matched 410
root/action pairs across 40 roots and produced 820 context records. The D1
audit then found 1,930 repeated pair-context comparisons with flip rate `0.0`
and stability score `1.0`. Preference stability is therefore structurally
estimable on the joined table. The audit nevertheless exits
`rejected_fail_closed`: the teacher is `offline_environment_cost_v1`,
`teacher_score = -real_task_cost` exactly, and 164 outcome/test records lack
complete predicted/realized latent references. This is a D1
structure/provenance repair, not D-JEPA teacher validation; D2 remains
blocked.

Artifacts:

- `results/gdd_wm_repeated_candidate_contexts_20261015/repeated_context_manifest.json`
- `results/gdd_wm_repeated_candidate_contexts_20261015/d1_preference_stability.json`
- `experiments/gdd_wm/build_repeated_candidate_contexts.py`

The D0 validator now also performs an early teacher/execution leakage
preflight. On the joined v5 RGB-reference table it keeps schema status as
`passed` but sets `teacher_execution_validation` to
`rejected_exact_negative_cost_leakage` and `d1_ready=false`, with 640 numeric
teacher/cost pairs detected. This prevents a leaked cost oracle from being
mistaken for a validated teacher at the D0-to-D1 boundary.

## Fresh non-PushT CUDA replication (two disjoint 32-episode panels per environment)

To extend the planning evidence without reusing the old `seed20261100` panel,
two completely disjoint fresh panels were run on the host RTX 5060 Ti. Both
use the frozen environment checkpoint, identical CEM settings
(`horizon=5`, `action_block=5`, `receding_horizon=5`, `64/8/8`), same-seed
RGB-only pairing against random control, and no outcome-based controller or
checkpoint selection. Reacher uses MuJoCo EGL rendering. The first panels
start at seeds `20261083`/`20261084`; the second at `20261120`; the resulting
64-episode aggregates contain no duplicate seeds.

| Environment | Episodes | Frozen LeWM | Random | Paired wins/losses/ties | Delta | Bootstrap 95% CI | Exact sign-test p |
|---|---:|---:|---:|---:|---:|---|---:|
| TwoRoom | 64 | 20.31% | 3.13% | 11/0/53 | `+0.171875` | `[+0.078125,+0.265625]` | `0.0004883` |
| Reacher | 64 | 48.44% | 7.81% | 28/2/34 | `+0.40625` | `[+0.265625,+0.53125]` | `4.34e-7` |

The fresh non-PushT evidence index is
`le-wm-main/results/cross_environment_replication_20261015/nonpusht_evidence_fresh64.json`.
It keeps the estimands separate: TwoRoom and Reacher provide scoped live RGB
planning consequences, Cube remains the strict `0/8` action-support negative
boundary, and the 32-pair PointMaze offline gate remains statistically
inconclusive. These fresh positives strengthen the environment-scoped
planning result but do not establish a universal world model, causal latent
dynamics, or a commutator certificate.

The raw and aggregate artifacts are:

- `le-wm-main/results/cross_environment_replication_20261015/tworoom_seed20261083/`;
- `le-wm-main/results/cross_environment_replication_20261015/tworoom_seed20261120/`;
- `le-wm-main/results/cross_environment_replication_20261015/reacher_seed20261084/`;
- `le-wm-main/results/cross_environment_replication_20261015/reacher_seed20261120/`;
- `le-wm-main/results/cross_environment_replication_20261015/tworoom_aggregate_fresh64/`;
- `le-wm-main/results/cross_environment_replication_20261015/reacher_aggregate_fresh64/`.

## External learned matched-quality benchmark (PointMaze + PushT)

The earlier coverage audit had used checkpoint records with missing external
fields. I reran the strict benchmark on the existing immutable external
evidence panels instead: 90 PushT snapshots with independent `+a/-a/zero`
action intervention and fresh RGB-only planning consequence, plus 50
PointMaze records with independent action-risk and held-out consequence. The
benchmark matches prediction MSE, latent dimension, predictor spectrum,
training stage, action distribution, and independent action risk before
computing the planning-risk contrast.

The resulting cross-environment panel has two identifiable environments and
three identifiable matched strata. Its primary estimand is only
`Delta R = +0.0030757`, bootstrap CI `[0, 0.0092272]`, and one-sided
sign-flip `p=0.5048` in the direct benchmark (`p=0.4935` in the
cross-environment wrapper); the positive gate is false. Leave-one-environment-out
therefore remains a diagnostic output, not a generalization result. The
strict result is a useful resolution of the previous “missing external
evidence” blocker, but it is also a direct negative result for the claim that
alignment/geometry provides a stable planning-risk increment after matching
the declared nuisance factors.

Artifacts:

- `le-wm-main/results/matched_learned_transfer_benchmark_external_crossenv_20261015/`;
- `le-wm-main/results/cross_environment_transfer_diagnostic_external_20261015/`;
- `le-wm-main/results/best_paper_readiness_audit_external_20261015/`;
- `le-wm-main/results/oral_experiment_coverage_audit_20261015/`.

The readiness tool still rejects learned action release because the complete
independent calibration interval and the predeclared multi-metric external
gain schema are absent. The current evidence supports a more precise claim:
the transfer-risk tool is implemented and fail-closed, but the universal
alignment selector is not empirically established.

## Reacher root-disjoint action-ranking consequence audit

To extend the external consequence beyond PushT, the frozen Reacher checkpoint
was evaluated with the same shared-candidate structure used by the TwoRoom
audit. The model receives only the rendered current RGB, rendered target RGB,
and action/history fields. A fixed 17-candidate macro-action bank is replayed
from the same root seeds for evaluator-only terminal qpos labels; simulator
state is never passed to the ranking function or used for selection. The run
uses CUDA/EGL and 32 root-disjoint roots:

`le-wm-main/results/reacher_shared_candidate_ranking_cuda_root32_20261015`

The held-out ranking result is negative: top-1 agreement is `0/32`, mean
pairwise action-ranking inversion is `0.5225` (bootstrap 95% CI
`[0.4782, 0.5643]`), and mean selected-candidate planning regret is `1.0706`
(`[0.7505, 1.4081]`) in terminal two-joint qpos error. Mean terminal
multistep task error is `1.8268` (`[1.5047, 2.1602]`). This is a useful
non-PushT failure localization: positive RGB success-rate panels do not imply
that the frozen model ranks a shared action bank correctly.

The result was connected to the strict external-consequence schema through
`results/external_independent_gain_coverage_reacher_20261015/`. Four metrics
are available (`action_ranking_inversion`, `multistep_task_error`,
`angle_closure`, and `planning_regret`); `contact_retention` is explicitly
blocked as not applicable to Reacher. The readiness audit therefore remains
fail-closed (`candidate_artifact_count=1`, `valid_artifact_count=0`). The
metric parser now requires available finite estimates and intervals for every
required field, root-disjoint held-out provenance, and an explicit independent
consequence flag; a metric-name-only artifact cannot open the Best-Paper gate.

The alternate predeclared target condition `(+0.65,+0.65)` was then run with
the same 32-root CUDA/EGL panel. It gives top-1 agreement `20/32`, inversion
`0.4380 [0.4074,0.4688]`, and planning regret
`0.0844 [0.0350,0.1571]`, whereas the original `(+0.65,-0.65)` condition gives
top-1 `0/32` and regret `1.0706 [0.7505,1.4081]`. The pooled 64-root artifact
is
`le-wm-main/results/reacher_shared_candidate_ranking_aggregate_64root_20261015.json`:
top-1 `31.25%`, inversion `0.4802 [0.4525,0.5086]`, and regret
`0.5775 [0.3749,0.7924]`. The correct conclusion is conditional transfer:
goal geometry changes the action-identifiability regime. This is neither a
universal Reacher positive nor a universal negative, and it strengthens the
need to condition any transfer-risk claim on task geometry and action support.

### Deployment-history mismatch repair

The first Reacher ranking audit used `history_len=1`, while the live Reacher
policy contract uses three RGB frames. This was a real protocol mismatch, so
the audit was rerun rather than folded into the previous result. With
`history_len=3`, the original `(+0.65,-0.65)` condition improves from top-1
`0/32` to `4/32`, inversion `0.4989 [0.4674,0.5303]`, and regret
`0.3977 [0.1714,0.6671]`; the `(+0.65,+0.65)` condition gives top-1 `23/32`
and regret `0.1112 [0.0247,0.2595]`. The deployment-matched pooled 64-root
artifact is
`le-wm-main/results/reacher_shared_candidate_ranking_aggregate_64root_history3_20261015.json`:
top-1 `42.19%`, inversion `0.4700 [0.4476,0.4928]`, and regret
`0.2544 [0.1190,0.4142]`. The runner default is now frozen to
`history_len=3`, and future Reacher ranking audits must override it explicitly
if they intend a different protocol.

## Reacher latent/action readout intervention and TwoRoom live consequence

The next intervention tested whether a calibration-only latent/action ridge
readout could repair the Reacher action-ranking boundary. The frozen encoder
and predictor, shared 17-candidate bank, history length 3, CUDA/EGL runtime,
and root-disjoint split were held fixed. Two target geometries were evaluated
with 16 calibration and 16 held-out roots each.

The direct readout overfit or damaged ranking on both geometries:

| Target delta | Held-out raw regret | Direct ranker regret | Held-out raw top-1 | Direct ranker top-1 |
|---|---:|---:|---:|---:|
| `[+0.65,-0.65]` | `0.3854` | `0.5588` | `0/16` | `3/16` |
| `[+0.65,+0.65]` | `0.0905` | `0.4219` | `13/16` | `5/16` |

A nested calibration intervention then selected a dimensionless blend with
`alpha=0` on both geometries, exactly reverting to the frozen raw cost. This
prevented harm but produced no held-out improvement. Adding low-dimensional
action-effect/desired-effect geometry summaries also failed to produce a
positive held-out consequence. The artifacts are:

- `le-wm-main/results/reacher_latent_action_ranker_v2_cuda_root32_target_neg_20261015.json`;
- `le-wm-main/results/reacher_latent_action_ranker_v2_cuda_root32_target_pos_20261015.json`;
- `le-wm-main/results/reacher_latent_action_ranker_effect_geometry_cuda_root32_target_neg_20261015.json`;
- `le-wm-main/results/reacher_latent_action_ranker_effect_geometry_cuda_root32_target_pos_20261015.json`.

The scientific diagnosis is now sharper: the frozen Reacher model contains
some geometry-conditional ranking signal, but a small calibration readout is
not a stable action-identifiability repair. This is retained as a negative
boundary rather than promoted into the online controller.

In contrast, an independent 32-episode TwoRoom RGB panel tested a previously
fitted ranker and two controls on the exact same seeds. Raw frozen Euclidean
utility achieved `14/32` successes versus random `3/32`; the latent/action
ranker achieved `10/32`, and the action-conditioned physical readout achieved
`8/32`. Paired against raw, the ranker delta was `-0.125` (5 losses, 1 win,
26 ties; bootstrap CI `[-0.28125,0]`) and the action-conditioned readout delta
was `-0.1875` (7 losses, 1 win, 24 ties; CI `[-0.34375,-0.03125]`). Thus the
offline TwoRoom replay-regret improvement does not transfer automatically to
closed-loop planning; raw candidate-bank LeWM remains the selected scoped
controller for this panel.

Artifacts and implementation:

- `le-wm-main/results/live_tworoom_ranker_validation_20261015/`;
- `le-wm-main/experiments/summarize_tworoom_method_comparison.py`;
- `le-wm-main/experiments/audit_reacher_latent_action_ranker.py`.

All runs used the host RTX 5060 Ti with `--device cuda`; the full regression
suite passes `720 passed, 3 skipped`. These results do not change the broader
claim boundaries: TwoRoom has a scoped RGB planning consequence, while a
universal action-transfer repair remains unestablished.

## Reacher action-support intervention and continuous-CEM control

The constant 17-candidate Reacher audit was then extended with a predeclared
shared sequence bank: 17 constant sequences, 80 single-pulse sequences, and
64 pulse-recovery sequences. On the same 32 roots and history-3 contract, the
offline evaluator shows a large action-support effect. For target
`[+0.65,-0.65]`, the oracle error falls from `0.7496` to `0.2137`, while the
model-selected terminal error falls from `1.0701` to `0.2188`; for
`[+0.65,+0.65]`, the selected error falls from `0.8624` to `0.2869`. The
support decomposition artifacts are:

- `le-wm-main/results/reacher_candidate_support_comparison_target_neg_20261015.json`;
- `le-wm-main/results/reacher_candidate_support_comparison_target_pos_20261015.json`.

This offline gain does not by itself repair the live planner. With the same
`256/8/16` CEM budget and identical fresh 32-seed panels, raw continuous CEM
was stronger than the full fixed bank:

| Target geometry | Random | Raw Gaussian CEM | Full 161-bank CEM | Compact 49-bank CEM |
|---|---:|---:|---:|---:|
| `[+0.65,-0.65]` | `1/32` | `21/32` | `14/32` | `20/32` |
| `[+0.65,+0.65]` | `1/32` | `7/32` | `1/32` | `1/32` |

The full bank consumes too much of the sampling budget and harms continuous
sequence exploration. The compact bank reduces the negative-geometry damage
(`-0.03125` versus raw, CI `[-0.1875,0.125]`) but does not improve the raw
baseline; on the positive geometry both bank variants are below raw CEM. The
paired method artifacts are:

- `le-wm-main/results/live_reacher_sequence_candidate_planning_20261015/method_comparison_neg.json`;
- `le-wm-main/results/live_reacher_sequence_candidate_planning_20261015/method_comparison_pos.json`;
- `le-wm-main/experiments/run_live_reacher_sequence_candidate_planning.py`;
- `le-wm-main/experiments/summarize_reacher_live_method_comparison.py`.

The resulting diagnosis is specific: fixed candidate support explains much of
the replay negative result, but the correct online remedy is support-aware
continuous exploration, not replacing CEM with a large fixed bank. Raw CEM
provides a scoped positive frozen-RGB Reacher consequence on both geometries;
the sequence-bank intervention is retained as a mechanism/control study and
does not establish universal latent dynamics or stable pose closure.

## Cube RGB/latent observability and one-step effect audit

To separate a visual-information bottleneck from an encoder bottleneck, I
added `experiments/audit_cube_rgb_observability.py`. It collects current/goal
RGB frames and a predeclared action, fits calibration-only ridge readouts, and
evaluates them on root-disjoint frames. The same protocol is applied to
low-resolution RGB and the frozen Cube LeWM embedding. Simulator state is used
only for offline labels; it is never a runtime policy input.

The label-free generic 16-direction support probe produced no block motion,
confirming that an approach schedule without contact coverage cannot diagnose
contact observability. A separate `guided_intervention` condition uses the
initial block-to-effector direction only to obtain mechanism samples. It is
explicitly not a planner and is not eligible as RGB-only planning evidence.
The formal CUDA/EGL artifact is:

`le-wm-main/results/cube_rgb_observability_20261015/guided_panel80x24/observability.json`.

It contains 80 roots, 1,920 transitions, root-disjoint 40/40 calibration/test
roots, and five task IDs. Physical block motion occurs in `10.31%` of frames;
the simulator gripper-contact flag is positive in only `2.29%`, so the latter
is retained as a diagnostic but is not treated as the primary effect label.
For the sparse `physical_motion` label, held-out balanced accuracy is:

| Feature family | RGB current+action | RGB pair+action | frozen latent current+action | frozen latent pair+action |
| --- | ---: | ---: | ---: | ---: |
| physical-motion classification | `0.756` | `0.751` | `0.779` | `0.562` |

The held-out next-block displacement RMSE for the same four families is,
respectively, `2.36e-4`, `4.43e-4`, `2.59e-4`, and `3.11e-4` m. Geometry
readouts are similarly weak and close between RGB and latent (for example,
block-to-goal XY RMSE is `0.0659` for RGB current and `0.0644` for frozen
latent current). These are mechanism diagnostics with sparse, privileged
collection coverage, not evidence of a repaired Cube planner.

The actionable diagnosis is therefore two-part: (i) generic RGB CEM does not
reach the contact-support region reliably, and (ii) after guided coverage,
one-step effect information is not stably preserved by goal-conditioned frozen
latent features. The next controller intervention must increase independent
contact-support coverage and use a fail-closed action-identifiability gate;
adding another unconstrained goal/readout head is not justified by this panel.

### RGB color-geometry control

As a lower-capacity visual control, the same audit now extracts the largest
red connected-component centroid from current and goal RGB, without using
simulator labels. The rerun is stored at
`le-wm-main/results/cube_rgb_observability_20261015/guided_panel80x24_color/observability.json`.
The color chart gives held-out block-to-goal XY RMSE `0.0642`, close to the
frozen-latent current readout `0.0636`; its sparse physical-motion balanced
accuracy is `0.50`, so the mask alone does not identify the action effect.
This is a useful negative control: visible object color/centroid geometry is
not sufficient to recover the contact-conditioned action effect, and the
next repair should target calibrated image-to-action geometry or richer
temporal interaction features rather than another static goal concatenation.

## Temporal action-effect gate: representation-conditioned replication

The next non-PushT intervention implements
`experiments/audit_cube_temporal_action_effect.py`. It collects consecutive
RGB frames, frozen LeWM embeddings, and the previously executed action, then
fits static, additive-temporal, and low-dimensional `innovation x action`
readouts on calibration roots only. A separate event/support residual gate
uses calibration-only quantiles and falls back to the static readout when the
event or support test fails. The evaluator uses simulator state only for the
next-block displacement label; no state enters a feature or action choice.

Two independent 80-root CUDA/EGL panels (40 held-out roots each) were run
under the same guided mechanism-coverage protocol. The root-level aggregate
is `results/cube_temporal_action_effect_20261015/aggregate_seed20261018_20261020.json`.
The central result is representation conditional:

| method | baseline | held-out RMSE delta | bootstrap 95% CI | wins/losses |
|---|---|---:|---|---:|
| RGB temporal | RGB static | `+1.769e-4` | `[+0.889e-4,+2.692e-4]` | `31/49` |
| RGB interaction | RGB static | `+1.460e-4` | `[+1.122e-4,+1.818e-4]` | `12/68` |
| latent temporal | latent static | `-1.006e-4` | `[-1.906e-4,-0.168e-4]` | `49/31` |
| latent interaction | latent static | `-0.057e-4` | `[-0.646e-4,+0.486e-4]` | `34/46` |
| RGB event/support gate | RGB static | `+0.435e-4` | `[+0.324e-4,+0.552e-4]` | `13/60` |

Thus additive temporal history gives a reproducible held-out effect-prediction
improvement only for this frozen latent representation, while RGB history and
the interaction lift are harmful or unstable. The event/support gate rejects
about 79% of test frames and still loses to its static fallback; it is not
admissible as an online controller. This is a sharper diagnosis of the
architecture/representation condition, not a stable planning repair.

The artifacts are:

- `results/cube_temporal_action_effect_20261015/guided_panel80x24_cuda/`;
- `results/cube_temporal_action_effect_20261015/guided_panel80x24_seed20261018_cuda/`;
- `results/cube_temporal_action_effect_20261015/guided_panel80x24_seed20261020_cuda/`;
- `experiments/audit_cube_temporal_action_effect.py`;
- `experiments/summarize_cube_temporal_action_effect.py`.

Claim boundary: this is root-disjoint offline mechanism evidence. It does not
establish causal latent dynamics, action ranking, frozen visual planning, or a
universal memory/gate. The next admissible test is an independent shared-
candidate action-ranking and closed-loop consequence panel; the latent
temporal readout may enter that test only as a predeclared intervention, not
as a post-hoc selected controller.

## Cube RGB-only support-gate closed-loop replication

The first independent closed-loop consequence exposed a concrete protocol
failure: the fixed descent/close prelude often left the controller outside the
contact-support region, so static and temporal rankers executed identical
actions and produced `16/16` ties. I therefore implemented a predeclared,
fail-closed support exploration rule in
`experiments/run_cube_temporal_closed_loop.py`: when the frozen ranker's
candidate-score spread is below the calibration-independent threshold
`5e-4 m`, the runtime cycles through `x+`, `x-`, `y+`, and `y-`. It does not
read simulator state and does not use test outcomes for threshold or policy
selection.

The rule was first run on the original 16-episode held-out panel and then on
an independent 32-episode panel with a new calibration/test seed. The paired
aggregation is written by
`experiments/summarize_cube_temporal_closed_loop.py` to
`results/cube_temporal_closed_loop_20261015/aggregate_support_gate_panels_48ep.json`.
Across all 48 paired episodes:

| comparison | wins | losses | ties | mean final-distance delta | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| temporal - static | `7` | `0` | `41` | `-0.01764 m` | `[-0.03380,-0.00489] m` |
| support-gate - static | `10` | `0` | `38` | `-0.02010 m` | `[-0.03670,-0.00712] m` |

The ordinary temporal arm has no loss but is heterogeneous: it has `0/16`
wins in the first panel and `7/32` wins in the independent panel. The support
fallback actually triggers on only `4/48` episodes; among those it produces
`3` improvements and `1` tie, with no deterioration. The other seven support-
gate improvements are the same temporal ranker improvement rather than a
fallback effect. Thus the result is a positive, but sparse and
support-conditioned, RGB-only Cube consequence—not evidence that the memory
method is universally effective.

The repaired diagnosis is now sharper: (i) the earlier all-tie negative result
was partly caused by failure to acquire action support, (ii) a fixed
fail-closed exploration cycle can unlock motion on a small subset of roots,
and (iii) temporal ranking remains panel-sensitive. This supports a scoped
action-support intervention and motivates replication on the existing
TwoRoom/Reacher/PointMaze protocols; it does not establish universal transfer,
causal latent dynamics, stable pose closure, or a universal commutator
certificate.

## Reacher support-bank repair boundary

The earlier Reacher intervention showed that a large fixed pulse/recovery bank
consumes too much of the continuous CEM budget. I tested a more conservative
repair in `experiments/run_live_reacher_sequence_candidate_planning.py`:
`support4` injects only four midpoint pulse-recovery candidates, and the new
`candidate_bank_iterations=1` option injects them only in the first CEM
iteration before restoring the full Gaussian budget. Both choices were frozen
before the fresh paired runs.

Against raw Gaussian CEM on identical 32-seed panels, repeated `support4`
obtained `13/32` versus raw `15/32` on target `[+0.65,-0.65]` (paired delta
`-0.0625`, `4/6/22` wins/losses/ties, bootstrap CI `[-0.25,0.125]`) and
`3/32` versus raw `9/32` on `[+0.65,+0.65]` (delta `-0.1875`, `0/6/26`, CI
`[-0.34375,-0.0625]`). The one-shot probe obtained `13/32` versus `15/32`
on the first geometry (delta `-0.0625`, `2/4/26`, CI `[-0.21875,0.09375]`)
and `10/32` versus `9/32` on the second (delta `+0.03125`, `5/4/23`, CI
`[-0.15625,0.21875]`).

This falsifies the proposed fixed support-bank repair as a general online
improvement: even a four-candidate probe can perturb the learned CEM update,
and the small positive result is target-geometry-specific with a CI crossing
zero. The negative causal diagnosis is now more precise than “candidate
coverage is missing”: support must be acquired without imposing a fixed action
prior on the CEM distribution. The formal controller remains raw continuous
CEM; `support4` and one-shot support are retained as failed interventions.
Artifacts:

- `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_neg.json`;
- `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_pos.json`;
- `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_once_neg.json`;
- `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_once_pos.json`.

## Executable finite-horizon transfer-risk theory audit

I ran the theorem and control-contract runners rather than leaving the theory
as prose. The artifacts are
`results/finite_horizon_transfer_risk_decomposition_20261015/` and
`results/covariant_transfer_risk_control_audit_20261015/`. The audit verifies
four logically separate requirements:

- the positive scalar tightness witness attains the decomposed bound exactly
  (`0.5904`);
- two realizations with identical spectrum, covariance, commutator,
  prediction MSE, and action-inverse risk have different `G_h` (`0.9` versus
  `0.4`) and different task risk (`0` versus `0.5`);
- equal spectral radius does not remove non-normal amplification: the
  finite-horizon bound is `0.68563` for the non-normal operator versus
  `0.23428` for the normal operator at the same residual envelope;
- the local nonlinear certificate reports `certified_local` with bound
  `0.07442`, while the control runner selects the fallback when action
  identifiability is false and only permits a candidate when all declared
  gates pass.

This is now a reproducible formal control theory artifact with theorem,
tightness example, explicit counterexamples, local applicability conditions,
and fail-closed action semantics. It remains synthetic theory evidence: no
learned checkpoint has yet supplied independently calibrated closure,
planner-reachable off-manifold, action-effect, readout, and nonlinear
remainder envelopes, so this run does not establish learned-model
transfer-risk prediction or a universal commutator certificate.

## GDD-WM D2 projection-only gate implementation

The previously missing D2 entry point is now implemented as
`experiments/gdd_wm/train_projection_distillation.py`. It accepts a
provenance-keyed feature table, freezes the dynamics, fits only a low-capacity
linear projection on train-root teacher pairs, and separately requires
prediction-preservation and independent planning-consequence JSON gates. It
does not accept candidate-set context, teacher rank, or held-out outcome as a
runtime feature.

Running it on the current repeated-context artifact produced
`results/gdd_wm_projection_geometry_distillation_20261015/d1_blocked/d2_projection_distillation.json`:
status `blocked_d1_gate`, with no projection training executed. The cause is
the already audited `rejected_fail_closed` D1 teacher (offline cost oracle,
exact negative-cost leakage, and incomplete runtime references). This is an
implementation closure and a verified fail-closed result, not D2 scientific
evidence. Once an independent D-JEPA-style teacher and complete feature table
arrive, the same entry point can run D2 without changing the gate semantics.

The D2 interface was hardened after this audit. A feature table must now carry
the schema `gdd_wm_student_feature_table.v0.1`, the exact SHA-256 of the
candidate dataset, the canonical record-key schema, and explicit declarations
that candidate-set context, teacher rank, and held-out outcome are not feature
inputs. It must also declare frozen dynamics. The projection rank is bounded by
both feature dimension and the number of train records. Existing tables without
this provenance are rejected before fitting; this is an intentional fail-closed
change, not a retroactive repair of the blocked D1 package.

## Cross-environment planning evidence index (non-PushT)

To prevent the research conclusion from being driven by PushT, I generated the
native-estimand evidence index
`results/cross_environment_planning_evidence_20261015/` from the frozen
TwoRoom, Reacher, Cube, and PointMaze artifacts. The rows are deliberately not
pooled because their protocols and estimands differ. TwoRoom live RGB paired
planning is positive (`5/0/19`, delta `+0.2083`, 95% CI `[+0.0417,+0.3750]`);
Reacher is positive (`16/0/8`, delta `+0.6667`, 95% CI `[+0.4583,+0.8333]`).
Cube remains a failure boundary (`0/0/8` wins/losses/ties, delta `0.0000`),
and the independent PointMaze gate is inconclusive (delta `+0.1007`, 95% CI
`[-0.0277,+0.2478]`, statistical gate false). This supports scoped
cross-environment evidence, not a universal world model, causal latent physics,
or commutator certificate.

## GDD-WM teacher eligibility audit (2026-10-15 continuation)

The read-only audit
`experiments/gdd_wm/audit_teacher_candidate_packages.py` was rerun after the
repeated-context construction. It inspected 24 candidate packages and found
`0/24` eligible. The common blockers are exact
`teacher_score = -real_task_cost` leakage, the `offline_environment_cost_v1`
teacher identity, no repeated root/candidate contexts in the original packages,
and incomplete predicted/realized runtime references on outcome/test rows. The
artifact is
`results/gdd_wm_teacher_candidate_audit_20261015/teacher_candidate_audit.json`.
This localizes the remaining negative result to teacher provenance and data
structure; it does not justify constructing a synthetic teacher or bypassing
D1 to obtain a positive D2 result.

## Non-leaking train-only readout teacher control (2026-10-15 continuation)

To separate D1 infrastructure from teacher quality, I constructed a declared
control with `experiments/gdd_wm/build_train_only_readout_teacher_package.py`.
It joins two independent checkpoint-stage contexts on exact root/action hashes,
fits a frozen standardized ridge readout on train roots only, and retains
complete RGB successor references for calibration/outcome rows. The resulting
package has 2 contexts, 40 roots, 640 matched root/action pairs, and 1,280
records. The package eligibility audit is positive (`1/1 eligible`), so the
repeated-context and runtime-reference gates are operational on a non-leaking
control.

The D1 result is not a scientific success: stability is only `0.6919` (flip
rate `0.3081`), aggregate execution consistency is `0.4719`, and held-out
outcome consistency is `0.5800`. The control therefore localizes the remaining
failure to the quality/decision validity of the latent readout teacher rather
than to missing D1 data structure. It is explicitly not a D-JEPA teacher and
does not authorize D2 or planning claims. Artifacts:

- `results/gdd_wm_nonleaking_teacher_control_20261015/package/`;
- `results/gdd_wm_nonleaking_teacher_control_20261015/package_audit/`;
- `results/gdd_wm_nonleaking_teacher_control_20261015/d1_preference_stability.json`.

### Predeclared feature-family intervention (2026-10-16 continuation)

I compared five families under the same two-context, train-only ridge protocol:
`predicted_terminal`, `source_plus_terminal`, `source_plus_delta`,
`source_delta_action`, and `action_summary`. No family was selected from the
outcome split. The root-bootstrap summary is
`results/gdd_wm_nonleaking_teacher_control_20261015/feature_family_panel/summary.json`.

The action-only control is the only family with a clear held-out movement in
this small panel: outcome pairwise consistency `0.6179` (95% CI
`[0.4882,0.7321]`) and mean regret `0.1060` (95% CI `[0.0139,0.2284]`),
versus predicted-terminal consistency `0.5625` (`[0.4647,0.6596]`) and regret
`0.2116` (`[0.1131,0.3054]`). The paired action-only improvement is
`+0.0554` consistency (CI `[+0.0050,+0.1019]`) and `-0.1055` regret (CI
`[-0.2011,-0.0230]`). By contrast, adding latent source/delta/action features
does not improve the held-out paired consistency (`-0.0620`, CI
`[-0.1975,+0.0879]` for `source_delta_action`).

This is not a latent-geometry success. The action-only baseline is a direct
action-distribution confound and cannot support D-JEPA, causal-latent, or D2
claims. The current negative result is sharper: the available PointMaze bank
contains decision signal in the candidate action summary, while the frozen
latent features do not yet add independently reliable preference information.

### Residualized action baseline intervention (2026-10-16 continuation)

To test the required conditional estimand directly, I added the
`residualize_action` mode to
`experiments/gdd_wm/build_train_only_readout_teacher_package.py`. For each
predeclared latent family, the teacher first fits an action-summary readout on
train roots, then fits the latent feature only to the train residual, and
evaluates the sum on the held-out outcome roots. The action baseline is never
fit or tuned on outcome labels.

Against the frozen action-only control (`0.6179` pairwise consistency,
`0.1060` regret), residualized predicted-terminal obtains `0.5932` consistency
and `0.1310` regret. Its paired consistency delta is `-0.0247` with root
bootstrap CI `[-0.0634,+0.0118]`; paired regret delta is `+0.0250` with CI
`[0.0000,+0.0736]`. Residualized source-plus-delta is also below the action
control (`0.5708` consistency), and residualized source-delta-action is lower
still (`0.5006`).

This is a direct negative result for the current target estimand:
the frozen latent does not yet provide reliable transfer information beyond
the action summary on this held-out PointMaze panel. The artifact is
`results/gdd_wm_nonleaking_teacher_control_20261015/residualized_feature_panel/summary_vs_action.json`.
It is a control diagnosis, not a D-JEPA or D2 result.

### Fixed-capacity CUDA audit of the latent residual (2026-10-16 continuation)

To test whether the preceding negative `Delta Risk | action summary` result was
only a linear-readout underfit, I ran the predeclared
`experiments/gdd_wm/audit_train_only_teacher_capacity.py` on host CUDA. The
protocol uses the same two checkpoint-stage contexts, exact root/action join,
`source_delta_action` features, train-root-only fitting, and held-out outcome
roots. It compares a standardized linear residual readout with one fixed
shallow MLP (`[64,32]`, SiLU, 800 AdamW epochs); no calibration or outcome
selection is performed.

The artifact is
`results/gdd_wm_nonleaking_teacher_control_20261015/capacity_audit/capacity_audit.json`.
It contains 1,280 records, 24 train roots, 8 outcome roots, and reports
`device=cuda`.

| readout | outcome consistency | mean regret | regret delta vs action-only |
| --- | ---: | ---: | ---: |
| action-only | 0.3821 | 0.2666 | 0 |
| linear residual | 0.4994 | 0.1098 | -0.1568 [-0.2602,-0.0678] |
| fixed MLP residual | 0.4212 | 0.2035 | -0.0631 [-0.1797,+0.0360] |

The linear residual has a held-out regret reduction in this small PointMaze
control, but consistency uncertainty still crosses zero. The fixed MLP does
not improve the linear residual and its regret interval crosses zero. Thus the
earlier negative result cannot be repaired by simply increasing readout
capacity; the current evidence points to a low-sample/support-sensitive
latent-action interface, with a possible approximately linear signal. This is
still a diagnostic control: it is not D-JEPA teacher evidence, D2 evidence,
or an independent closed-loop planning consequence.

The first direct-script invocation exposed that this package must be launched
from the repository root as a module (`.venv/bin/python -m
experiments.gdd_wm.audit_train_only_teacher_capacity ...`), otherwise Python
does not add the project root and `experiments` cannot be imported. The
module entry point was used for the verified CUDA run; no CPU fallback was
used.

### Coordinate and estimand controls for the capacity diagnosis (2026-10-16)

The same CUDA audit was rerun on two independent seed contexts using the
canonical goal-frame candidate bank. With the original raw action summary,
the linear residual regret delta was `-0.0305` (95% CI
`[-0.0829,+0.0322]`) and the MLP delta was `-0.0271`
(`[-0.1297,+0.0664]`). The apparent `-0.1568` raw-action gain from the first
control therefore did not replicate after the candidate context was aligned.

I then added two explicit protocol controls to the runner:
`--action-frame goal_relative` and `--target-mode root_centered`. The
goal-relative absolute-cost run gave linear regret delta `0.0000` and MLP
delta `-0.0265` (CI `[-0.1184,+0.0389]`). The stricter root-centered,
goal-relative run gave linear regret delta `0.0000` and MLP delta `-0.0360`
(CI `[-0.1836,+0.0723]`). The linear consistency changes were small and do
not establish a planning-relevant gain.

These controls localize the earlier positive point estimate to a raw-action /
root-cost confound rather than a robust latent increment. The immediate repair
is methodological: use canonical action coordinates and root-centered
within-candidate estimands before interpreting any latent residual. The current
capacity audit consequently remains negative for an independent latent
transfer contribution; no D2 or teacher claim is opened.

Artifacts:

- `results/gdd_wm_nonleaking_teacher_control_20261015/canonical_goalframe_capacity_audit/`;
- `results/gdd_wm_nonleaking_teacher_control_20261015/canonical_goalframe_capacity_audit_goal_relative/`;
- `results/gdd_wm_nonleaking_teacher_control_20261015/canonical_goalframe_capacity_audit_root_centered/`.

### Context-safe cross-environment capacity replication (2026-10-16 continuation)

The preceding capacity audit contained a grouping bug: candidates from different
contexts could enter the same root-level ranking group. The runner now keeps
each `context x root` cell separate, macro-averages repeated contexts within
unique roots, and bootstraps unique outcome roots. The old mixed-context JSON is
therefore not final evidence. The corrected CUDA artifacts are:

- `results/gdd_wm_cross_environment_20261016/tworoom_capacity_root_centered_context_safe/`;
- `results/gdd_wm_cross_environment_20261016/reacher_capacity_root_centered_context_safe/`.

Both audits use two history-conditioned contexts, 16 context-root outcome
cells, 8 unique outcome roots, train-root-only fitting, and the same fixed
`source_delta_action`/action-only protocol. The target is root-centered, while
the action frame is explicitly recorded as raw; this is not yet the canonical
goal-relative learned benchmark.

| environment | readout | mean regret delta vs action-only | 95% root-bootstrap CI |
| --- | --- | ---: | ---: |
| TwoRoom | linear residual | `-7.1922` | `[-19.6837,+6.2154]` |
| TwoRoom | fixed MLP residual | `-16.8480` | `[-32.0209,-0.4949]` |
| Reacher | linear residual | `-1.6113` | `[-2.0892,-1.1483]` |
| Reacher | fixed MLP residual | `-1.9058` | `[-2.3356,-1.4063]` |

This repairs the earlier negative result enough to show that the latent
residual is not uniformly useless: a fixed-capacity residual improves held-out
within-root regret in Reacher and in the fixed MLP arm of TwoRoom. The TwoRoom
linear interval still crosses zero, and both panels are only history/context
replications rather than independent architectures or seeds. The result is
therefore a cross-environment capacity-diagnostic signal, not evidence for a
universal latent transfer certificate, D-JEPA teacher, D2 distillation, or
closed-loop planning gain. The next confirmatory run must use the canonical
goal-relative action frame and independent architecture/seed strata.

### Canonical goal-relative recheck (2026-10-16 continuation)

The original non-PushT panels did not contain evaluator-side source/goal
coordinates, so `goal_relative` could not be evaluated. The TwoRoom and Reacher
panel exporters now write `source_state_raw` and `goal_raw` as evaluator-only
fields under panel schema `gdd_wm_live_candidate_panel.v0.2`; these fields are
never passed to RGB model ranking. New panels were generated on CUDA/EGL at:

- `results/gdd_wm_cross_environment_20261016/tworoom_history{1,3}_goalframe/`;
- `results/gdd_wm_cross_environment_20261016/reacher_history{1,3}_goalframe/`.

The canonical `--action-frame goal_relative --target-mode root_centered`
capacity audits then gave:

| environment | readout | mean regret delta vs action-only | 95% root-bootstrap CI |
| --- | --- | ---: | ---: |
| TwoRoom | linear residual | `-3.0036` | `[-11.5527,+3.6907]` |
| TwoRoom | fixed MLP residual | `-3.3582` | `[-18.7137,+13.9097]` |
| Reacher | linear residual | `+0.1087` | `[-0.2508,+0.4948]` |
| Reacher | fixed MLP residual | `+0.2677` | `[-0.1120,+0.6630]` |

The raw-action root-centered gains therefore do not survive the canonical
action frame: TwoRoom becomes inconclusive and Reacher moves in the wrong
direction, with all canonical intervals crossing zero. The negative result is
now localized to a measurement/interface confound rather than a missing MLP
capacity fix. The corrected protocol is the only admissible one for future
claims; no latent independent transfer gain, D2 teacher, or planning claim is
opened by the raw-action result.

### Frozen RGB goal-latent conditioning intervention (2026-10-16 continuation)

The canonical residual interface was strengthened without using privileged
state: the panel exporters now retain the frozen RGB goal embedding and the
capacity audit exposes `--goal-conditioning latent_relative`. The added
features are `goal-source` and `predicted-goal`, alongside the existing
source/delta/action features; model ranking and all outcome labels are
unchanged. New CUDA/EGL panels are in:

- `results/gdd_wm_cross_environment_20261016/tworoom_history{1,3}_goal_latent/`;
- `results/gdd_wm_cross_environment_20261016/reacher_history{1,3}_goal_latent/`.

With the canonical goal-relative/root-centered protocol, adding frozen RGB
goal-latent geometry gives:

| environment | readout | mean regret delta vs action-only | 95% root-bootstrap CI |
| --- | --- | ---: | ---: |
| TwoRoom | linear residual | `-4.3478` | `[-11.5736,+0.5659]` |
| TwoRoom | fixed MLP residual | `-1.2867` | `[-13.9894,+15.0122]` |
| Reacher | linear residual | `+0.2943` | `[+0.0060,+0.6155]` |
| Reacher | fixed MLP residual | `-0.2280` | `[-0.7132,+0.2270]` |

Goal-latent conditioning improves some pairwise-consistency estimates but does
not produce a stable held-out regret reduction across the two environments.
The Reacher linear arm is measurably harmful, and the MLP interval crosses zero.
This rules out missing goal embedding as the sole cause of the negative result.
The current failure is therefore an action-effect/support problem in the frozen
latent interface, not a readout-capacity or goal-feature omission. The new
feature is retained as a predeclared diagnostic arm, not promoted to a
controller or transfer certificate.

### Capacity-direction repair and calibration-safe fallback (2026-10-16 continuation)

The capacity runner contained a consequential direction bug: its target is
terminal task cost (lower is better), but root-level candidate selection used
`argmax`. That operation selected the predicted worst candidate and could
manufacture negative regret results. The runner now uses `argmin`, and a
regression test explicitly checks that a lower predicted cost is selected.
Artifacts produced before this repair are retained for provenance but are not
treated as final capacity evidence.

After the direction repair, the canonical goal-relative/root-centered
comparison changed materially. With the same frozen panels and CUDA protocol,
the uncalibrated linear/MLP residual arms gave TwoRoom regret deltas of
`+15.5624`/`+6.6561` versus action-only, while Reacher gave
`-0.8184`/`-1.0049`. The result is therefore not a universal readout-capacity
failure: it is environment-conditional and potentially harmful on TwoRoom.

The runner now uses the pre-existing calibration roots to select a fixed ridge
strength from `{0.01, 0.1, 1, 10, 100, 1000}` and computes a paired root
bootstrap interval against action-only. A residual is enabled only if the
calibration interval's upper endpoint is strictly below zero; otherwise the
protocol falls back to action-only. This is a fail-closed diagnostic gate, not
outcome tuning.

| environment | calibration decision | held-out action-only regret | held-out gated regret | gated delta (95% CI) |
| --- | --- | ---: | ---: | ---: |
| TwoRoom | fallback (`CI [-6.57,+8.07]`) | `2.0465` | `2.0465` | `0.0000 [0.0000,0.0000]` |
| Reacher | enable residual (`CI [-1.40,-0.33]`) | `1.9437` | `1.5323` | `-0.4114 [-0.7901,-0.0288]` |

The safe gate removes the confirmed TwoRoom negative transfer and preserves a
held-out Reacher gain, but it does not establish a universal latent-transfer
claim. The correct scientific interpretation is now sharper: calibration
uncertainty is itself a useful transfer-risk signal, and support-sensitive
latent residuals must be rejected when their independent calibration benefit is
not statistically separated from zero. The gate remains an offline diagnostic
until an independent RGB closed-loop consequence is run under the same frozen
rule.

Artifacts:

- `results/gdd_wm_cross_environment_20261016/tworoom_capacity_safe_gate/`;
- `results/gdd_wm_cross_environment_20261016/reacher_capacity_safe_gate/`;
- `experiments/gdd_wm/audit_train_only_teacher_capacity.py`;
- `tests/test_gdd_wm_train_only_teacher_capacity.py`.
