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
