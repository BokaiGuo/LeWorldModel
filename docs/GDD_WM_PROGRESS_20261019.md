# GDD-WM execution progress — 2026-10-19

## What is now executable

The workstream now has a real, provenance-separated control path:

```text
repeated candidate contexts
  -> train-root-only frozen predictive teacher
  -> D1 stability/execution audit
  -> projection-only student diagnostic
  -> independent RGB planning consequence (separate gate)
```

The new teacher is `frozen_predicted_latent_ridge_v1`. It fits a ridge readout
from predicted terminal latent + goal coordinates to **training-root** task
costs, freezes the parameters, and scores calibration/outcome candidates
without reading their realized costs. It is an execution-independent teacher
control, not a validated D-JEPA teacher.

## First execution

Artifact root:

`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_cuda/`

- 820 records, 40 roots, 2 repeated candidate-set contexts;
- D1 repeated-context stability: `1.000` (1,930 repeated candidate-pair entries);
- teacher/execution pair consistency: `0.580` overall and `0.622` on outcome rows;
- no exact `teacher_score = -real_task_cost` leakage;
- strict successor-reference support remains incomplete (`164` outcome/test rows),
  so this first control was not released as final teacher evidence;
- projection diagnostic outcome regret: `0.1056` versus frozen baseline `0.1111`;
- calibration regret increased from `0.1764` to `0.1862`, so the result is not
  a uniformly positive distillation result.

The D2 output is therefore `completed_diagnostic` with
`rejected_missing_independent_gate`; it does not authorize a planning claim.

## Strict successor repair and geometry follow-up

The missing-reference count was caused by artifact plumbing rather than absent
successor data. Three validated context-0 RGB successor reports (train,
calibration, outcome) were merged with strict key and array checks; context 1
already had a complete report. The resulting strict package is:

`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_strict_cuda/`

- 640 + 640 RGB successor rows merged and joined;
- `missing_runtime_ref_count=0` under the strict D1 audit;
- `audit_teacher_candidate_packages.py` reports one eligible package;
- D1 remains execution-consistent but imperfect: `0.580` overall and `0.622`
  on outcome pairs.

The first projection negative was then traced to two implementation choices:

1. the original student feature table used an arbitrary zero goal anchor;
2. the RGB consequence evaluator failed to subtract the frozen goal anchor.

Both are repaired. A train-only teacher-best global anchor is now supported,
and the evaluator reads the exact frozen `goal_features` from the feature
table. The repaired projection gives outcome regret `0.0666`, but its centered
baseline is also `0.0666`; therefore the result is an absolute repair, not an
incremental GDD gain. A predeclared rank sweep (`1,2,4,8`) found no rank that
beats the centered baseline on both calibration and outcome. The fail-closed
interpretation is retained: the anchor fixes the geometry bug, but the current
projection has not yet demonstrated independent decision value.

## Independent RGB consequence

After CUDA preflight passed on RTX 5060 Ti, the episode-disjoint PointMaze RGB
planning run completed at:

`le-wm-main/results/gdd_wm_pointmaze_rgb_consequence_20261019_cuda`

This is a separate frozen-LeWM risk-gating consequence, not evidence that the
GDD projection itself improves closed-loop planning:

| method | success | final distance | fallback |
| --- | ---: | ---: | ---: |
| visual dynamics | `0.60` | `0.325` | `0.00` |
| wm RGB | `0.10` | `1.077` | `0.00` |
| wm RGB guarded | `0.65` | `0.324` | `0.795` |

The guarded arm is a positive risk-gating result, but it does not close the
GDD-WM teacher-to-student-to-planner chain. The next hard gate is to supply
independent realized successor references and connect the frozen student
geometry to a planner without candidate IDs, teacher ranks, or privileged
state.

## Engineering repair

`train_projection_distillation.py` now accepts the existing nested
repairability panel schema only when at least one nested environment audit is
explicitly `permitted_predicted_dynamics`; authorization is not pooled across
environments. Regression coverage was added for this contract.

## Claim boundary

The current evidence supports a reproducible predictive-teacher control and a
diagnostic projection result. It does **not** yet support: D-JEPA teacher
correctness, universal preference distillation, GDD-specific closed-loop gain,
or a universal world-model claim.

## Goal-conditioned teacher intervention and live consequence — 2026-10-19

The missing `goal_latent_model` field in the strict PointMaze package was not
filled synthetically.  Instead, the pre-existing root-disjoint CUDA/EGL
TwoRoom and Reacher goal-latent panels were used:

```text
results/gdd_wm_cross_environment_20261016/
  tworoom_history{1,3}_goal_latent/panel/
  reacher_history{1,3}_goal_latent/panel/
```

Both panels contain 544 records with train/calibration/outcome root separation
and a valid frozen goal-latent reference.  The audit used CUDA, goal-relative
action summaries, root-centered targets, and `latent_relative` conditioning;
all fitting remained train-root-only.

The intervention identified two distinct failure modes.  TwoRoom still failed
the calibration gate: adding predicted dynamics increased held-out regret and
the safe policy was action-only fallback.  Reacher showed a local offline
signal, but the goal-relative branch is evaluator-only and therefore was not
exported as a runtime controller.  A deployable raw-action artifact was then
fit under the same train-only protocol and passed its calibration gate.  Its
predicted-dynamics cross-fit selection was frozen before the live consequence;
the outcome was not used for selection.

The independent live consequence used the frozen Reacher checkpoint, rendered
current/goal RGB only at runtime, the same 64 seeds, and the fixed `capacity17`
candidate bank within the same CEM budget.  The comparison was:

| controller | success | mean final qpos distance |
| --- | ---: | ---: |
| frozen LeWM baseline | 35/64 | 0.17378 |
| train-only capacity readout | 38/64 | 0.16488 |

The paired success-rate difference is `+0.0469` with bootstrap 95% CI
`[-0.0156, +0.1250]`; exact paired McNemar `p=0.375`.  Paired final-distance
difference is `-0.00890` with bootstrap 95% CI `[-0.02289, +0.00360]`.  The
readout changed the bank argmin on `672/776` evaluated rows (`86.6%`) and did
not abstain for low support.  Thus the consequence is a useful positive
direction and a deployment-path check, but not a statistically established
GDD-WM gain.  The artifact remains an intermediate, non-oral result until an
independent predeclared seed panel and cross-environment replication establish
the paired consequence.

Artifacts:

- `results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_goalrelative_latentrelative/`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_goalrelative_latentrelative/`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_raw/`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_live_baseline_64/`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_live_capacity_64/`

The paired statistics are serialized by
`experiments/gdd_wm/summarize_live_teacher_consequence.py` at
`results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_live_capacity_64/paired_summary.json`.

The current actionable repair is therefore not to relax the gate or claim
universal transfer.  It is to increase independent root/architecture coverage
and require the same frozen teacher-to-runtime artifact to pass both Reacher
and a non-Reacher environment before promoting the GDD-WM claim.

## Independent Reacher confirmation and TwoRoom local-ranker adjudication — 2026-10-20

A new root-disjoint Reacher confirmation panel was run on CUDA before looking
at its outcomes.  Teacher and baseline used the same 64 new seeds, frozen
checkpoint, `capacity17` bank, CEM budget, horizon, and target.  The teacher
reached `38/64`; the frozen baseline reached `34/64`.  There were four
teacher-only wins and zero baseline-only wins, for a paired delta of `+0.0625`
and exact McNemar `p=0.125`.  This is an independent positive-direction
confirmation of the earlier Reacher result, not yet a statistically established
claim; the predeclared aggregate remains the primary estimate.

The machine-readable run is
`le-wm-main/results/gdd_wm_reacher_teacher_confirmation64_20261020_cuda/`.

The proposed execution-before-action local-effect ranker was also tested on a
fresh, same-seed TwoRoom panel.  The comparison was made correctly between two
hybrid arms, so the shared RGB fallback cannot be counted as teacher gain:

| controller | success |
| --- | ---: |
| teacher + hybrid local ranker | `18/32` |
| baseline + hybrid local ranker | `19/32` |

The paired delta was `-1/32`, with exact McNemar `p=1.0`.  The negative result
is mechanistically informative: the local chart was usable only after several
observations, while the door-phase fail-closed gate blocked the ranker on a
large fraction of calls.  Thus this intervention did not provide an early
teacher decision advantage, and it must not be presented as a positive GDD-WM
result.

During this audit, hybrid runtime diagnostics were found to be silently
exported as `null`.  `RiskGatedRGBPolicy.runtime_diagnostics()` now forwards
the wrapped solver diagnostics and the runner uses policy-level diagnostics;
the repair is covered by the TwoRoom targeted suite.  Future local-ranker
panels can therefore distinguish a true no-op from a support/phase abstention.

## Active-probe failure boundary — 2026-10-20

To test whether the TwoRoom failure was primarily an early-support problem, a
predeclared PointMaze active-probe arm executed four RGB cardinal pulses,
fit an episode-local action-effect Jacobian, and then planned with the local
chart.  On 20 held-out roots it reached `20/20`, equal to the visual-dynamics
baseline, but had worse mean final distance (`0.1688` versus `0.1485`).  The
fixed information-action cost therefore erased the local-chart benefit.

A second predeclared arm triggered the probes only when the frozen GDD and
visual-dynamics proposals disagreed.  It triggered on `85%` of roots and
reached only `15/20`, versus `20/20` for visual-dynamics; mean final distance
was `0.5448`.  Post-action verification was high (`0.996`), so the failure is
not an RGB measurement or verification bug.  The disagreement signal itself
is not a valid uncertainty signal for this checkpoint: it authorizes probes
exactly where the GDD proposal is harmful.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_jacobian_20261020_cuda_fresh20/planning_report.json`
- `le-wm-main/results/gdd_wm_pointmaze_gdd_disagreement_verified_local_20261020_cuda_fresh20.json`

This closes the proposed “probe earlier” repair as a no-go for the current
GDD student.  The safe repair is a train/calibration-only teacher authorization
gate: if the teacher does not beat the RGB dynamics arm on calibration under
the declared task consequence and support constraints, the deployment policy
must fail closed to visual dynamics.  Such a gate can prevent negative
transfer, but it cannot be reported as an independent GDD planning gain.

## Calibration-only teacher authorization — 2026-10-20

The fail-closed authorization rule is now implemented in
`run_live_pointmaze_planning.py`.  On explicitly disjoint calibration roots,
the teacher must satisfy both:

1. success rate at least as high as visual-dynamics; and
2. mean final distance no higher than visual-dynamics.

Otherwise the teacher is rejected before held-out execution and the held-out
method runs as visual-dynamics, with the report retaining the requested method
name and an explicit `gdd_authorization_fallback` marker.

On a fresh 10-root authorization split, the disagreement-triggered GDD teacher
scored `8/10` with mean final distance `0.4233`; visual-dynamics scored `10/10`
with mean final distance `0.1491`.  The authorization was therefore
`false`.  On the separate 20-root held-out split, the automatic fallback
produced `20/20` and mean final distance `0.1603`, exactly matching the
visual-dynamics arm, with zero active probes.  This is a validated negative
transfer prevention result, not a GDD planning gain.

Artifact:
`le-wm-main/results/gdd_wm_pointmaze_gdd_calibrated_authorization_20261020_cuda_fresh20.json`.
The authorization helper has dedicated regression tests, and the full
PointMaze targeted suite passes.

## Cross-bank GDD guard repair — 2026-10-20

Code audit found that the former `gdd_wm_rgb_guarded` gate compared the GDD
proposal only with the best action inside the GDD-restricted candidate family.
That could certify a bad action simply because the visual-dynamics reference
action was not present in the comparison set.  The gate now compares against
the actual visual-dynamics fixed bank and releases a GDD action only when its
calibration dynamics cost is within the declared tolerance of that bank.

On a fresh 20-root diagnostic panel, the repaired guard achieved the same
`20/20` success as visual-dynamics and improved mean final distance by
`0.00389` (`0.15644` versus `0.16034`).  The paired distance direction was
heterogeneous (`6` teacher-better roots versus `10` visual-better roots), and
success was tied on every root.  Therefore this is a valid reduction of
negative transfer, not an independently established teacher gain.

Artifact:
`le-wm-main/results/gdd_wm_pointmaze_gdd_guarded_crossbank_diagnostic_20261020_cuda_fresh20.json`.

A stricter zero-regret diagnostic was also run with tolerance `0.0`.  It
increased the fallback rate to `98.6%` but produced a slightly worse mean
final distance (`0.16182`) than visual-dynamics (`0.16034`), with the same
`20/20` success.  This rules out “allowing a small negative dynamics margin”
as the sole cause of the remaining variation: simply tightening the gate does
not create teacher information.  The current positive direction is therefore
best treated as a conservative support repair, not as evidence that the GDD
geometry is task-sufficient.

## Support-projected GDD branch — 2026-10-20

The route/action-support repair was implemented as a separate
`gdd_wm_rgb_projected_guarded` method.  It projects a GDD proposal toward the
actual visual-dynamics action using the existing calibration-supported segment
projection, and rejects projections below the predeclared minimum alpha.

On the same fresh 20-root diagnostic panel it matched the cross-bank hard gate
exactly: success `20/20` and mean final distance `0.15644` versus `0.16034` for
visual-dynamics.  The projection alpha took only values `0` or `1`, so no
continuous intermediate action was actually supported by the current dynamics
chart.  This identifies a further boundary: the current learned/action chart
does not expose a useful local interpolation path; the repair is a safety
wrapper, not a new teacher signal.

Artifact:
`le-wm-main/results/gdd_wm_pointmaze_gdd_projected_guarded_v2_20261020_cuda_fresh20.json`.

## Terminal-stability gate follow-up — 2026-10-20

The two 128-root large-support panels isolated a concrete failure mode:
the source-action readout improved minimum distance (short-term reachability)
but increased terminal distance.  The runtime intervention therefore gained a
calibration-only terminal-stability gate.  It allows the train-only readout to
reorder only supported candidates inside the frozen LeWM base-cost 50th
percentile; rows with fewer than two stable supported candidates fail closed.
The gate uses no privileged qpos and no held-out outcomes.

Implementation and regression coverage:

- `le-wm-main/experiments/capacity_readout_runtime.py`
- `le-wm-main/tests/test_capacity_readout_runtime.py`
- targeted regression: `35 passed`

A third root-disjoint CUDA panel was then run with the same `capacity17`,
`best` final-action, CEM, checkpoint, and action-support protocol.  The frozen
baseline was rerun on the identical 128 roots before adjudication:

| controller | success | mean final qpos distance | mean minimum qpos distance |
| --- | ---: | ---: | ---: |
| frozen LeWM baseline | 63/128 | 0.18279 | 0.08869 |
| source-action readout + terminal gate | 71/128 | 0.16650 | 0.08527 |

The paired intervention-minus-baseline estimates are:

- success: `+0.0625`, bootstrap 95% CI `[0.0000, 0.1250]`, readout-only wins
  `13` versus baseline-only wins `5`;
- final distance: `-0.01629`, CI `[-0.03573, 0.00299]`;
- minimum distance: `-0.00342`, CI `[-0.01062, 0.00380]`.

The new panel is a positive directional replication and removes the earlier
terminal-distance sign reversal, but the continuous terminal-distance interval
still crosses zero.  It is therefore not yet an established independent
GDD-WM teacher gain.  The gate was active on 74 planner rows, changed the
within-support argmin on all 74, and did not abstain for low support.

The existing 384-root aggregate must not be read as a single-policy estimate:
the first two batches used the older ungated artifact behavior, while the
third used the new 50th-percentile terminal gate.  It is retained only as a
heterogeneous replication ledger.  The current supported insight is narrower:

> Separating short-horizon reachability from frozen-model terminal stability
> can remove the direction of negative terminal transfer on a fresh Reacher
> panel, but does not yet establish cross-environment teacher-to-planner gain.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_independent_stability_gate_128_best_cuda/`
- `le-wm-main/results/gdd_wm_reacher_independent_stability_gate_128_best_cuda_baseline/`
- `le-wm-main/results/gdd_wm_reacher_independent_stability_gate_128_best_cuda/paired_comparison.json`
- `le-wm-main/results/gdd_wm_reacher_capacity_pooled_384_cuda/aggregate.json`

## Cross-environment gate audit: TwoRoom — 2026-10-20

The same terminal gate was intentionally tested outside Reacher rather than
assuming that its frozen-cost proxy transfers.  On a fresh 64-root TwoRoom
CUDA panel, using the predeclared `vision_door16` bank and the frozen
goal-conditioned bilinear artifact:

| controller | success |
| --- | ---: |
| frozen LeWM baseline | 23/64 |
| bilinear teacher + base-cost 50th-percentile gate | 20/64 |

The paired delta was `-3/64` (`-0.046875`), with exact McNemar `p=0.375`.
The gate abstained on 114 of 912 evaluated rows.  This is a genuine
cross-environment negative result, not a runtime crash: frozen base cost is
not a valid terminal-stability proxy during TwoRoom's door phase.  The
Reacher repair is therefore not promoted as a universal gate.

The phase-aware fail-closed arm was then evaluated on a new root-disjoint
64-root panel.  It uses RGB reactive control during the detected door phase
and leaves the teacher available only after the phase transition:

| controller | success |
| --- | ---: |
| frozen LeWM baseline | 12/64 |
| direct bilinear teacher + gate | 11/64 |
| phase-aware hybrid teacher/RGB fallback | 26/64 |
| RGB reactive | 26/64 |

The phase-aware hybrid was a recovery mechanism, not an independent teacher
gain: it tied the RGB reactive ceiling and exceeded the frozen baseline only
by a non-significant point estimate.  Its paired teacher comparison remains
open, and it must not be reported as GDD-WM planning improvement.

The runtime now records the bilinear stability gate separately from the MLP
gate and has a regression test for the invariant that unsupported or
out-of-quantile candidates remain on the frozen cost.  The current method
boundary is consequently sharper:

> A terminal-stability gate based on frozen predictive cost can repair
> source-action readout drift in Reacher, but the same proxy fails at
> discrete contact/door transitions.  Cross-environment deployment requires
> phase-specific support and an RGB/observed-effect fallback.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_terminal_gate_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_terminal_gate_fresh64_20261020_cuda/paired_summary.json`
- `le-wm-main/results/gdd_wm_tworoom_phase_gate_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_phase_gate_fresh64_20261020_cuda/paired_summary.json`

## Verified-local effect repair follow-up — 2026-10-20

Because the direct bilinear gate failed at the TwoRoom door transition, the
next intervention combined three fixed rules: RGB phase detection, a local
RGB action-effect map estimated from short probes, and held-out effect
verification before releasing the inverse action.  The frozen bilinear
teacher remained unchanged; this is a verified-local repair arm, not a new
teacher.

On a fresh 64-root CUDA panel with identical `vision_door16` support:

| hybrid arm | success |
| --- | ---: |
| frozen baseline + verified local fallback | 21/64 |
| bilinear teacher + verified local fallback | 24/64 |
| RGB reactive ceiling | 25/64 |

The paired teacher-versus-baseline hybrid delta was `+3/64`, with one
baseline-only win and four intervention-only wins; exact McNemar `p=0.375`.
The direct teacher arms were `12/64` and `16/64` respectively, but this
point estimate is not a sufficient independent planning claim.  The local
policy was released 731 times and rejected by held-out effect verification
422 times, confirming that the verification gate is active rather than a
privileged-state oracle.

The repair therefore improves the direction and makes the failure mechanism
operationally visible, but it does not yet close the statistical gate.  The
stronger prior four-panel local-repair aggregate remains the main evidence
for a scoped TwoRoom recovery result; this new panel is an independent
replication with non-significant uncertainty.

Artifact:

- `le-wm-main/results/gdd_wm_tworoom_verified_local_gate_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_verified_local_gate_fresh64_20261020_cuda_baseline/`

## Runtime contract repair and closed-loop adjudication

The first two GDD live runs exposed two independent deployment mismatches:

1. The frozen RGB successor export uses the declared
   `raw / 255 -> transform -> *255 -> EncPredWM.encode` preprocessing round
   trip.  The live branch initially called `encode(raw)` directly.  A CUDA
   single-root audit showed the direct PCA source projection had mean absolute
   magnitude `4.048`, while the stored source chart has mean absolute magnitude
   `0.113`; the repaired contract reduced the live value to `0.132`.
2. The adapter was trained on the exact `paired_order_v2` 16-candidate family
   (two ordered ten-step pulses plus the deterministic seed-7000 support tail).
   The live branch initially supplied 32 repeated single-action pulses.  The
   replicated helper now matches the stored episode-1800 bank with maximum
   absolute error `0.0`.

The repairs are implemented in
`le-wm-main/experiments/run_live_pointmaze_planning.py` and covered by the live
PointMaze tests.  All runs below used CUDA, the same frozen checkpoint, the
same calibration roots, the same 20 held-out roots, and the same renderer.

| artifact | GDD policy | success | mean final distance | interpretation |
| --- | --- | ---: | ---: | --- |
| `gdd_wm_pointmaze_gdd_closed_loop_20261019_cuda` | original runtime chart | `0/20` | `1.289` | invalidated by source/action contract mismatch |
| `gdd_wm_pointmaze_gdd_closed_loop_20261019_cuda_v3` | source preprocessing repaired | `0/20` | `2.412` | source chart fixed, action chart still mismatched |
| `gdd_wm_pointmaze_gdd_closed_loop_20261019_cuda_v4` | preprocessing + exact paired bank | `1/20` | `2.410` | standalone frozen projection remains non-planning |
| `gdd_wm_pointmaze_gdd_closed_loop_20261019_cuda_v6` | fail-closed, tolerance `0.1` | `10/20` | `0.363` | safety gate removes most harmful GDD actions but trails RGB dynamics |
| `gdd_wm_pointmaze_gdd_closed_loop_20261019_cuda_v7_tol0` | fail-closed, tolerance `0.0` | `12/20` | `0.261` | near-baseline safety boundary; not an independent GDD gain |

The paired baseline in the same v6/v7 panel is visual dynamics: `14/20`, mean
final distance `0.276`.  GDD standalone therefore remains a negative result;
the guarded branch is a risk-recovery mechanism, not evidence of decision
distillation gain.  The guard rejects `96.2%` of GDD decisions at tolerance
zero and `92.0%` at tolerance `0.1`, which is itself evidence that the frozen
projection is not action-identifiable enough for release on this deployment
support.

The current GDD-specific failure boundary is now concrete: predictive-teacher
stability and RGB source-chart compatibility do not imply within-support
action ranking or closed-loop planning.  The next admissible GDD experiment is
not threshold tuning on these test roots; it is a new train/calibration-only
adapter trained on the exact runtime action family, followed by a fresh
root-disjoint held-out panel.

## Frozen-teacher failure-mode audit

The failure mode is now encoded in the read-only audit
`le-wm-main/experiments/gdd_wm/audit_frozen_teacher_failure_modes.py`, with the
report at
`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_strict_cuda/failure_mode_audit.json`.
It compares candidates only within their decision cells and never authorizes a
planner.  On the strict package:

- within-cell teacher/execution consistency is `0.580` (2,224/3,834 pairs);
- both repeated contexts are also `0.580`, so context duplication is not the
  cause;
- the horizon is fixed at `h=4`, so horizon mixing is not the cause;
- root-level consistency is heterogeneous (approximately `0.18` to `0.84`);
- action-support width has near-zero correlation with absolute teacher
  residual (`-0.029`), so a simple support-width threshold cannot repair it;
- teacher cost correlation is only `0.468`.

This changes the next training target.  The teacher must be fit and evaluated
as a root-local, task-conditioned action-effect ranker, with support and
uncertainty gates selected on calibration roots.  A global terminal-cost ridge
readout is retained as the control, not as the proposed solution.  Any new
closed-loop run must use fresh root-disjoint roots and report fallback rate;
the current held-out roots must not be reused for gate tuning.

## Root-local capacity repair diagnostic

The existing capacity audit initially could not run on this strict package:
its listwise loss assumed equal candidate counts in every train cell, while
the real package contains variable-size support cells.  The listwise loss in
`le-wm-main/experiments/gdd_wm/audit_train_only_teacher_capacity.py` now keeps
each cell at its native size and computes the soft ranking loss cell by cell.
The change preserves the train-only fit and does not pad, drop, or relabel
candidates.

The resulting diagnostic is stored at
`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_strict_cuda/root_local_capacity_cpu_diagnostic/capacity_audit.json`.
It is explicitly a CPU diagnostic because CUDA was not visible to the current
process.  It finds a promising but not yet admissible mechanism:

- calibration bilinear full predicted-dynamics versus no-dynamics regret delta:
  mean `-0.095`, 95% CI `[-0.173, -0.023]`;
- outcome regret point estimate: `0.092` for bilinear full versus `0.123` for
  action-only;
- strict five-fold root-disjoint train cross-fit full-vs-no-dynamics delta:
  mean `+0.010`, 95% CI `[-0.018, 0.043]`;
- the cross-fit gate therefore selects `action_only`, not the bilinear teacher.

This separates a calibration overfit from a validated repair.  The next
admissible teacher run must increase root-disjoint training support and repeat
the same cross-fit gate on CUDA before any closed-loop deployment.  The current
outcome point estimate is retained as diagnostic evidence only.

## Online local-effect repair path

The frozen teacher failure is now connected to an explicit recovery mechanism
in `le-wm-main/experiments/run_live_pointmaze_planning.py`:
`effect_adapter_rgb_online` and its fail-closed variant
`effect_adapter_rgb_online_htsc`.  After each executed pulse, the controller
computes the next position from the next RGB frame and updates a small local
ridge residual using only `(action, observed RGB displacement)`.  Candidate
actions are then ranked by the frozen effect policy plus this observed residual.
The implementation has no simulator-state policy input and starts with a zero
residual, so the frozen effect policy remains the episode-start baseline.
The residual is state-action conditioned on the RGB-estimated current velocity,
including velocity-action interaction terms, so the same action can receive
different corrections in different motion phases.
To prevent early high-variance updates, it stays disabled until four observed
pulses, applies sample-count shrinkage, and clips the task-effect correction to
a fixed norm trust region.  These are predeclared stability controls, not
held-out performance thresholds.
It now also records the selected candidate's predicted successor against the
next RGB-observed displacement; the local correction is enabled only after two
such verification errors are below the fixed `0.25` task-space threshold.
New effect-policy artifacts now replace that fallback with a threshold frozen
from the train/calibration-only predicted-versus-realized effect P90; legacy
artifacts remain backward compatible with the `0.25` fallback and are not
silently reinterpreted.

A CPU diagnostic artifact was generated from the existing 54 train-root / 14
calibration-root renderer-matched package at
`le-wm-main/results/htsc_preprocessing_repair_20261016/online_effect_policy_renderer_matched_p90_cpu_diagnostic/effect_policy.pt`.
Its frozen P90 verification threshold is `1.16848`.  This is not a planning
result and is not a reason to relax the trust region: the high threshold shows
that calibration error is broad, so release still requires the support gate,
successor verification gate, and the fixed correction norm cap together.
The persisted calibration distribution has 224 rows, mean `0.716`, P50
`0.650`, P90 `1.168`, P95 `1.615`, and max `1.738`.  This makes the remaining
problem explicit: the effect predictor is usable only as a coarse gate, not as
a precise successor certificate.

The directional audit is worse: effect-direction cosine has mean `-0.150`,
P50 `-0.297`, and P10 `-0.973`.  The new artifact builder therefore permits a
directional verification gate only when both its calibration mean and P10 are
positive.  The current package fails that gate and falls back to norm-only
verification; this prevents a seemingly sophisticated directional correction
from amplifying a systematically reversed action effect.

This is intentionally an online observed-effect method, not evidence that the
frozen GDD teacher is correct.  It is the admissible way to test whether the
remaining failure is recoverable by local system identification rather than by
more global latent distillation.  Its success, fallback rate, probe count, and
angle/task consequence must be measured on fresh roots against the same RGB
dynamics baseline.

The larger existing 100-root PointMaze capacity diagnostic also does not yet
authorize the teacher: goal-relative bilinear full dynamics reached a
root-disjoint cross-fit point estimate of `-0.0089`, but its 95% CI was
`[-0.0209, 0.0026]`; the standardized architecture pair gave `-0.0043` with
CI `[-0.0149, 0.0063]`.  Both therefore remain fail-closed.  This rules out
declaring a universal latent-geometry gain from the outcome point estimates.

## Runtime action-effect follow-up

The repository already contained a train/calibration-only
`runtime_repeated_pulse_v1` effect artifact:

`le-wm-main/results/htsc_preprocessing_repair_20261016/online_effect_policy_renderer_matched/effect_policy.pt`

and a rank-32 task-effect checkpoint trained on the same 32-action family.  I
connected both paths to the live runner and evaluated them on the same held-out
20-root panel.

| artifact | policy | success | mean final distance |
| --- | --- | ---: | ---: |
| `gdd_wm_pointmaze_runtime_task_effect_20261019_cuda_v1` | exact runtime task-effect adapter | `0/20` | `2.826` |
| `gdd_wm_pointmaze_verified_effect_20261019_cuda_v1` | renderer-matched observed-effect policy, RGB-maze waypoints | `8/20` | `0.624` |
| `gdd_wm_pointmaze_verified_effect_20261019_cuda_direct_v1` | same policy, global-direct goal diagnostic | `3/20` | `1.069` |

The paired visual-dynamics references were `14/20` (`0.276`) for the RGB-maze
protocol and `3/20` (`0.944`) for the direct-goal protocol.  Thus the existing
verified-effect artifact is a useful recovery/control path but does not provide
an independent GDD gain on this panel.  Changing local waypoints to the global
training chart also failed, so the remaining cause is not merely waypoint
covariate shift: the frozen task/effect readout still does not identify the
action-to-task consequence reliably enough for release.

This follow-up prevents an invalid positive interpretation: exact action-family
matching alone does not repair planning, and observed-effect verification must
be evaluated against a matched RGB-dynamics baseline rather than treated as an
automatic improvement.

## Effect-coordinate contract audit

The negative directional verification result was audited before changing any
sign convention.  The read-only audit is
`le-wm-main/experiments/gdd_wm/audit_effect_coordinate_contract.py`, and its
artifact is
`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_strict_cuda/effect_coordinate_contract_audit.json`.
It resolves terminal states through the original candidate-bank provenance,
checks the declared goal-relative action chart, and reports identity, axis
reflection, and axis-swap diagnostics without selecting a transformed chart.

The result does **not** support a global pixel/state or action-axis sign bug:

- raw mean-action/effect cosine on calibration rows: `-0.088`;
- identity chart over all non-degenerate rows: mean `0.166`, positive fraction
  `0.611`;
- no axis reflection or swap gives a coherent global improvement;
- the same calibration raw cosine becomes `0.909` after subtracting the
  within-root near-zero-action displacement baseline, with `0.987` positive
  rows.

The current negative directional signal is therefore primarily explained by
moving-state inertia and root-local dynamics contaminating raw terminal
displacement, not by a justified coordinate repair.  The audit also shows that
the raw signal is split-dependent (`-0.088` calibration versus `0.352`
outcome), so it cannot be used as a frozen action-effect certificate.  The
methodological repair is to use observed-effect residuals relative to a local
baseline and retain the existing fail-closed verification gate; no sign flip
is applied to the held-out artifact.  This is a diagnostic explanation, not a
claim that the frozen teacher has become planning-sufficient.

## Root-local effect-ranker diagnostic

The global teacher failure was tested against a more specific mechanism rather
than another global network.  The new read-only diagnostic is
`le-wm-main/experiments/gdd_wm/audit_root_local_effect_ranker.py`, with the
probe-4 artifact at
`le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261019_strict_cuda/root_local_effect_ranker_probe4_cpu_diagnostic.json`.

For each root, four candidates are selected by greedy farthest-point coverage
in the action basis, without reading task outcomes.  Their realized effects
fit a local ridge action-to-XY map, which ranks the remaining candidates.  On
the strict 40-root package:

| split | local minus probe-intercept regret | local minus frozen-teacher regret |
| --- | ---: | ---: |
| calibration (8 roots) | `-0.1110 [-0.1859,-0.0395]` | `-0.1572 [-0.2441,-0.0807]` |
| outcome (8 roots) | `-0.0865 [-0.1407,-0.0329]` | `-0.0933 [-0.2078,-0.0033]` |

The pooled local regret is `0.0060`, versus `0.1187` for the probe-intercept
control and `0.1704` for the frozen predictive teacher.  This is the first
positive evidence that the failure is root-local action-effect variation, and
that a small amount of observed intervention can repair ranking.  It is not a
closed-loop result: the diagnostic uses realized terminal effects for the
declared probes, so the next admissible experiment must replace these labels
with actual short-pulse RGB observations and compare against visual dynamics
on fresh roots.  The result therefore supports a local verified-transfer
hypothesis, not a universal GDD teacher claim.

## Online residual double-counting repair

The coordinate audit exposed a second implementation-level failure mode in the
online recovery path.  `OnlineLocalEffectResidual` previously stored the full
RGB-observed displacement and then added that quantity to the frozen effect
prediction.  This double-counted the frozen prediction.  It also updated the
residual after a visual-dynamics fallback, even though no frozen-effect
prediction had been executed for that observation.

The runtime contract is now repaired in
`le-wm-main/experiments/run_live_pointmaze_planning.py`:

- a frozen-effect execution stores `observed_effect - predicted_effect`;
- a visual-dynamics fallback does not update the frozen-effect residual;
- the existing support, verification, shrinkage, and trust-region gates are
  unchanged.

The behavior is centralized in `update_verified_execution()` so that the
fallback-abstention rule is tested directly rather than relying only on the
live-loop branch.

The same helper now accepts the RGB-dynamics no-action displacement as a
verification baseline.  Direction checks use
`(predicted - baseline, observed - baseline)`, while residual fitting still
stores the full `observed - predicted` correction.  This separates inertial
motion from action-dependent effect without changing the frozen policy or
reading simulator state in the controller.  The runtime obtains the baseline
from its calibration-only visual dynamics model.

This is a code-level correction that makes the online adaptation mathematically
consistent; it is not yet a closed-loop result.  It must be evaluated on a
fresh root-disjoint GPU panel against visual dynamics, frozen effect, online
residual, and HTSC fallback arms.  The new behavior is covered by the targeted
PointMaze regression suite (`25 passed`).

## Exact runtime action-chart gate

The existing online effect artifact exposed another source of negative results:
its protocol was trained on `paired_order_v2 / world_fixed`, while the live
`effect_policy_candidate_scores()` interface constructs
`runtime_repeated_pulse_v1 / goal_relative` candidates.  This silently made
the frozen effect model out-of-support even when its latent chart was valid.

The policy builder now records and validates both fields and rejects a
training/calibration bank that does not exactly match the live chart.  The
formal fresh-root runner is
`le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_cuda.sh`; its
first command is the real CUDA preflight, followed by exact-chart policy
training and the four-arm comparison:

`visual_dynamics`, `effect_adapter_rgb`, `effect_adapter_rgb_online`, and
`effect_adapter_rgb_online_htsc`.

The script was dry-run checked in the current sandbox and correctly stopped at
CUDA preflight rather than falling back to CPU.  Host `nvidia-smi` sees an RTX
5060 Ti, but the platform still blocks Python CUDA device-node access;
therefore no formal closed-loop result is claimed yet.

The exact-chart builder was also exercised as a CPU-only data/logic diagnostic
after the CUDA preflight failed (not as a scientific training result):
`le-wm-main/results/gdd_wm_online_local_effect_20261019_cpu_chart_diagnostic/`.
It confirms the runtime bank passes the new chart contract, with 54 train roots
and 14 calibration roots.  Importantly, the calibration direction gate still
rejects (`mean cosine -0.028`, `P10 -0.982`, P90 norm error `1.309`).  Thus
action-chart mismatch was a real deployment bug, but exact chart matching alone
does not solve the remaining directional non-identifiability; the online
observed-effect residual and fail-closed fallback remain necessary.

## Exact runtime-chart teacher re-audit

The exact deployment family was audited separately so that the earlier
paired-order/runtime-chart mismatch cannot be hidden inside the teacher result.
The train-only frozen predictive package is stored at
`le-wm-main/results/gdd_wm_runtime_teacher_20261019_cpu_diagnostic/` and uses
the runtime `runtime_repeated_pulse_v1 / goal_relative` chart, 54 train roots,
14 calibration roots, and 32 outcome roots.

The global teacher is not eligible for deployment on this chart:

- teacher/execution pair consistency is `0.635` across `47,596` pairs;
- teacher-cost correlation is only `0.049`;
- the package has one candidate context, so no cross-context teacher stability
  claim is possible;
- action-support width has near-zero correlation with absolute teacher residual
  (`-0.020`).

The corresponding root-local capacity audit is at
`root_local_capacity_single_context_cpu_diagnostic/capacity_audit.json`.  It is
explicitly a single-context CPU diagnostic, not a release gate.  On 32
root-disjoint outcome roots, the action-only root-local baseline has mean regret
`0.01278` (95% CI `[0.00381, 0.02491]`).  Adding predicted dynamics produces
the same mean regret and fails the calibration gate; the frozen selector
therefore falls back to action-only.  This localizes the teacher failure:
runtime latent/predicted-dynamics features provide no validated incremental
within-root action-effect information, while local action support is useful.

This changes the admissible recovery path.  A new frozen GDD teacher is not
authorized by increasing predictor capacity or tuning a projection.  The next
valid intervention is root-local observed-effect verification: execute a short
pulse, re-observe RGB, update the local action-effect model, and compare
against the same visual-dynamics and frozen-effect controls on fresh roots.  A
formal CUDA run remains required for closed-loop evidence; the current CPU
artifacts are diagnostic only.

## History-contract repair

The first history-2 re-audit initially reused the history-1 `arrays.npz`; its
SHA256 was identical, so that run was invalid as a memory comparison.  The
candidate bank was regenerated from the history-2 source export at
`le-wm-main/results/htsc_preprocessing_repair_20261019/runtime_pulse_history2_bank_rank32_renderer_development_rebuilt_cpu_diagnostic/`.
The regenerated arrays have a distinct SHA256 and the metadata contract is
now checked by `build_frozen_predictive_teacher_package.py`.

The repaired history-2 teacher still reports essentially the same result:
pair consistency `0.6353` and teacher-cost correlation `0.049`.  This is not
evidence that history is useless.  The package explicitly declares
`predictor_history_windows=1`, so the predicted terminal latent used by the
teacher is still produced by a single-window predictor.  The new package
metadata records this mismatch instead of silently calling it a history-2
teacher.  A genuine history-memory intervention therefore requires a
history-trained predictor or an online observed-history update, followed by
the same held-out planning gate.

## Genuine history-4 teacher and deployment truncation repair

The repository contains a separate, genuinely history-aware checkpoint panel:
`jepa_history4_{linear,mlp}`, with
`predictor_history_windows=4`.  Its train-only predictive teacher was audited
at `le-wm-main/results/gdd_wm_history4_teacher_20261019_cpu_diagnostic/`.
The teacher-cost correlation is `0.402` on this separate history-4 panel, but
execution consistency is only `0.455`; this shows that the history-aware panel
contains predictive signal, while not establishing a matched causal gain over
the runtime history-1 chart.  History alone does not make the teacher
action-identifiable.

The root-local observed-effect diagnostic on the same 100-root panel is:

| ranker | mean regret |
| --- | ---: |
| frozen history-4 teacher | `0.2166` |
| probe-intercept | `0.03735` |
| observed root-local effect | `0.01069` |

This is still execution-supervised and not closed-loop evidence, but it
isolates the recoverable mechanism: history improves prediction, while actual
local effect verification is what repairs action ranking.

The live planner previously hard-coded a two-frame context, which would silently
truncate a history-4 effect policy.  `run_live_pointmaze_planning.py` now reads
the loaded policy's declared history window, performs explicit no-op warm-up
blocks when needed, retains the full causal RGB window, and records
`context_windows`/`warmup_blocks` in every episode report.  The corresponding
test and package-provenance checks pass (`40` targeted tests).

The executable repair protocol is
`le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_history2_cuda.sh`.
It regenerates the history-2 bank after CUDA preflight, fits only on train
roots, calibrates on the declared calibration roots, and evaluates the same
fresh 20-root four-arm panel.  It refuses to run when CUDA is unavailable and
does not reuse the invalidated history-2 arrays.

The exact runtime history-4 policy was also fit as a CPU logic diagnostic at
`le-wm-main/results/htsc_preprocessing_repair_20261019/runtime_history4_exact_chart_cpu_diagnostic/`.
Its calibration regret bound is `0.471`, its directional gate is rejected, and
its verification P90 is `0.868`; therefore it is not released as a controller.
The formal fresh-root protocol is
`le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_history4_cuda.sh`.
It regenerates the bank with the genuine history-4 checkpoint, fits only on
train roots, calibrates on the declared calibration roots, and then evaluates
the four-arm RGB panel.  It fails closed before any work if CUDA is not
visible.

## Exact runtime history-4 effect audit

To separate predictor history from the deployment chart, a genuine history-4
checkpoint was used to regenerate the exact
`runtime_repeated_pulse_v1 / goal_relative` bank.  The self-contained teacher
and root-local audit are at
`le-wm-main/results/gdd_wm_runtime_history4_teacher_20261019_cpu_diagnostic/`.
The global history-4 teacher has execution consistency `0.656` and
teacher-cost correlation `0.358`.  The execution-supervised root-local ranking
diagnostic gives:

| ranker | mean regret |
| --- | ---: |
| frozen history-4 teacher | `0.2071` |
| probe-intercept | `0.2979` |
| observed root-local effect | `0.0220` |

This is the closest offline evidence so far to the actual runtime chart: local
observed effect still repairs the ranking, while the frozen teacher does not.
It remains a diagnostic and does not authorize a closed-loop claim.

The MLP-versus-linear rerun produced identical policy calibration metrics.  This
is expected after auditing `fit_pointmaze_online_effect_policy.py`: that policy
fits source-history plus action-basis features and realized intervention
effects; it does not consume predicted terminal latents.  Therefore this
architecture comparison cannot be presented as a GDD predictor gain.  GDD
teacher quality and online local-effect recovery remain separate estimands.

## Active information-gathering probe arm

The prior online residual arm updated only after an action selected by the
frozen effect policy. When its support or direction gate rejected, the
controller could fall back to visual dynamics without collecting action-space
information; this explains why offline local-effect gains did not necessarily
carry into closed loop.

`run_live_pointmaze_planning.py` now exposes a separate
`effect_adapter_rgb_online_probe` arm. Before ordinary planning it executes a
predeclared four-pulse cardinal probe set (`[+direct,-direct,+perpendicular,
-perpendicular]` at magnitude `0.5`) in the exact goal-relative runtime chart,
re-observes RGB after every pulse, and updates the local residual using
observed-minus-predicted effect with the calibrated no-action baseline. Probe
actions are not selected from task outcomes and are reported separately as
`active_probe_trace`; their cost is visible rather than hidden in the planning
score. The history-4 formal runner includes this arm alongside visual
dynamics, frozen effect, online local, and HTSC.

The probe generator is now shared with the planner's goal-relative action
chart: it rotates the four pulses into the current RGB-estimated
direct/perpendicular frame, so the intervention itself is inside the declared
runtime support.  A unit test checks magnitude, opposition, and orthogonality;
the targeted regression suite now has `41` passing tests.

The standalone support audit is
`le-wm-main/results/gdd_wm_runtime_history4_teacher_20261019_cpu_diagnostic/active_probe_support_audit.json`.
Across 100 roots and 400 probes, the mean action-support distance is `0.00184`,
P90 is `0.00604`, and `99%` of probes pass the frozen action-support threshold
(`0.01644`).  Thus the active-probe intervention does not introduce a new
action-chart mismatch; the remaining uncertainty is the observed RGB effect,
which must be tested in the fresh-root closed loop.

This is a proposed recovery intervention, not yet a positive closed-loop
result. Its release criterion is fresh-root success/progress improvement
against the same visual-dynamics baseline while reporting probe cost,
verification error, fallback rate, and residual trust.

## CUDA fresh-root adjudication and representation-contract repair

The formal history-4 CUDA run first exposed a deployment-contract bug rather
than a scientific failure.  The policy was fit from
`jepa_history4_mlp/seed_8411/stage_005.pt`, while the live planner loaded
`data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar`.  The resulting live
history projection was outside the policy's train support (mean support
distance approximately `47.9` versus threshold `1.247`), and the frozen
effect model produced implausible effects of order `20` in a task coordinate
whose observed pulses are order `0.1`.  Consequently every fail-closed online
gate rejected the effect policy.  This run is retained as a negative
contract-mismatch artifact, not as a teacher failure.

The shared planner path also had an uninitialized residual variable for the
non-online baselines.  That regression was fixed by explicitly initializing
the residual to `None` before dispatch; the targeted PointMaze suite is now
`32/32` passing.

The deployment-aligned history-1 policy was then evaluated on the identical
20 fresh test roots with the same frozen LeWM checkpoint and RGB-only input:

| method | success | mean final distance | effect release/fallback |
| --- | ---: | ---: | ---: |
| visual dynamics | `20/20` | `0.1601` | n/a |
| frozen effect, strict gate `0.25` | `16/20` | `0.3244` | release `0%` |
| online local, strict gate `0.25` | `16/20` | `0.2358` | release `0%` |
| online HTSC, strict gate `0.25` | `20/20` | `0.1601` | fallback `100%` |
| active probe, strict gate `0.25` | `20/20` | `0.1417` | fallback `100%` |

The strict gate is correctly conservative because the calibration-only regret
bound is `0.4322`, above the pre-existing tolerance `0.25`.  A predeclared
sensitivity at tolerance `0.5` permits the calibrated bound while preserving
the support gate.  On the same unopened roots it gives:

| method | success | mean final distance | release/fallback |
| --- | ---: | ---: | ---: |
| visual dynamics | `20/20` | `0.1601` | n/a |
| frozen effect | `16/20` | `0.3244` | release `79.4%` |
| online local | `16/20` | `0.2358` | release `82.5%` |
| online HTSC | `16/20` | `0.2364` | fallback `15.3%` |
| active probe | `19/20` | `0.2007` | fallback `19.8%` |

This is a positive recovery relative to the frozen effect controller
(`16/20 -> 19/20`) and demonstrates that observed RGB probing can repair part
of the teacher-to-action-transfer gap.  It is not an independent gain over
the RGB visual-dynamics baseline (`20/20`), so it does not establish GDD-WM
closed-loop superiority.  The correct current claim is: representation
matching is necessary, and verified local effect probing improves a weak
frozen effect controller, but the LeWM teacher has not yet supplied an
independent planning advantage.

## History-4 matched-contract re-audit

The first attempt to rebuild history-4 with the standard deployment checkpoint
still failed. Two additional deployment-contract bugs were isolated with a
single-root CUDA audit:

1. A history-4 export uses source frames `5,10,15,20`, but the live planner
   warmed up only three no-op blocks and supplied `0,5,10,15`.
2. The live effect-policy branch called `encode(raw)` while the frozen export
   used `raw/255 -> transform -> *255 -> encode`. On episode 1800, the direct
   history-4 projection had support distance `47.67`; the wrapped projection
   had support distance `0.0727`.

Both are now fixed and covered by the live planner tests. The final matched
history-4 CUDA fresh-20 panel is:

| method | success | mean final distance | release/fallback |
| --- | ---: | ---: | ---: |
| visual dynamics | `20/20` | `0.1633` | n/a |
| frozen effect | `16/20` | `0.2901` | release `56.6%` |
| online local | `16/20` | `0.2930` | release `60.0%` |
| online HTSC | `20/20` | `0.1656` | fallback `40.7%` |
| active probe | `18/20` | `0.1920` | fallback `39.1%` |

This is a genuine recovery from the prior `0%` release / `4/20` frozen-effect
artifact caused by the chart mismatch. It still does not demonstrate an
independent GDD gain because the RGB visual-dynamics baseline also reaches
`20/20`. The remaining scientific bottleneck is therefore not interface
compatibility; it is whether the released effect predictions can improve a
strong RGB reactive controller without relying on the same visual-dynamics
fallback.

## Probe-verified local release follow-up

The history-4 matched panel was extended with a separate
`effect_adapter_rgb_online_verified_local` arm. This arm executes four
predeclared cardinal RGB probes, updates the local observed-effect residual,
and may release the local effect ranker only when the current state/action
remain inside support and the episode-local verification object is trusted.
The global frozen-teacher regret bound is not relabeled as passed; this is an
explicit test of the `probe -> observe -> local update -> plan` recovery path.

The CUDA fresh-20 result is:

| method | success | mean final distance | fallback rate | interpretation |
| --- | ---: | ---: | ---: | --- |
| visual dynamics | `20/20` | `0.1633` | `0.000` | paired RGB baseline |
| verified local history-4 | `18/20` | `0.2005` | `0.400` | local release occurred, no independent gain |

The local arm did not solve the negative result. It improves over the weak
frozen-effect controller (`16/20`) only in the weak-controller comparison,
while remaining below the paired visual baseline. History-2 gives the same
boundary more sharply: the frozen/online arms reached `2/20`, HTSC reached
`20/20` only through `100%` fallback, and its calibration P90 verification
threshold was `0.7893` with a negative lower-tail directional signal.

This narrows the cause: online probes can produce a trusted local residual,
but the current residual model is not sufficiently task-conditioned to replace
a strong RGB visual-dynamics controller. The new arm is covered by the live
runner contract and the matched history-4 reproducibility script. It remains
a recovery diagnostic, not a GDD planning success claim; the next admissible
repair is phase/waypoint-conditioned local modeling on a non-saturated
environment, not lowering the global bound or counting fallback success.

## Low-dimensional local-Jacobian and TwoRoom cross-environment checks

The high-dimensional residual was replaced by a separate diagnostic arm that
fits only an affine bias plus a 2-D action Jacobian from RGB-observed effects.
On the same history-4 PointMaze fresh-20 roots, the ungated local Jacobian
reached `12/20`, confirming that algebraic identifiability alone is unsafe.
Adding the predeclared task-conditioned consistency gate restored `20/20` and
reduced mean final distance from the visual baseline `0.1633` to `0.1352`, but
the local arm was used on only `8.1%` of decisions and the paired 95% bootstrap
interval for the distance gain was `[-0.0109, +0.0686]`. This is a promising
non-degradation signal, not a confirmed independent gain.

The remaining unused PointMaze roots `1888--1899` gave `12/12` success and
mean final distance `0.1051` versus `0.1195` for visual dynamics, but the
interval was still wide and the local arm started farther from the goal because
of its probes. It is retained as a small external check only.

The frozen TwoRoom phase-local effect-map protocol was also rerun on 32 fresh
CUDA roots. Direction gate and effect-map gate were both `14/32`, with paired
comparison `0/0/32`. Thus phase-local RGB effect maps do not yet add planning
value in TwoRoom; direct inverse-map replacement is harmful (`0/10/22` in the
existing 32-root audit). The cross-environment repair is therefore explicitly
fail-closed: use local effect maps to veto conflicting actions, never to
overwrite a planner action without task-level waypoint verification.
## Goal-latent runtime contract repair and Reacher v4 consequence

The first repaired goal-conditioned artifact exposed a second deployment
failure: the train-only goal-aware residual produced a cost in its own
regression units, while unsupported Gaussian CEM proposals retained the
frozen LeWM cost. Directly mixing these values caused incomparable-scale
ranking and the first live v2 run fell to `27/64`.

The runtime now preserves only the supported-bank ordering and maps it
rank-affinely onto the frozen cost scale before merging with off-bank
proposals. This is a deployment-contract repair, not a test-outcome-based
weight change, and is covered by the capacity runtime regression suite.

The repaired CUDA Reacher consequence was rerun on the unchanged 64-seed
panel:

| controller | success | mean final qpos distance |
| --- | ---: | ---: |
| frozen LeWM baseline | `35/64` | `0.17378` |
| goal-aware rank-affine readout | `38/64` | `0.14763` |

The paired success difference is `+3/64` (`+0.0469`), with exact McNemar
`p=0.375` and bootstrap 95% CI `[-0.0156, +0.1250]`. The paired final
distance improvement is `-0.02616`, with bootstrap 95% CI
`[-0.06289, -0.00191]`. Runtime instrumentation confirms that the readout
was active: it evaluated `776` supported candidate rows and changed the
within-bank argmin on `679` rows (`87.5%`).

TwoRoom remains unchanged at `3/8` in the single-process CUDA smoke panel;
there is no cross-environment replication yet. Therefore the strongest
current claim is a Reacher-scoped, train-only goal-aware readout with a
positive held-out distance consequence after fixing a real runtime contract
bug. It is not yet evidence for a statistically established GDD-WM gain,
teacher validity across environments, or universal decision distillation.

Artifacts:

- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_raw_latentrelative_v2/capacity_readout.npz`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/reacher_live_capacity_latentrelative_v4_rankaffine_trace_64/`
- `results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_capacity_latentrelative_v2_smoke8/`

The runtime repair is implemented in
`le-wm-main/experiments/capacity_readout_runtime.py`; the contract test is
`le-wm-main/tests/test_capacity_readout_runtime.py`.

## TwoRoom cross-environment consequence

The same repaired artifact was evaluated on a fresh 32-episode TwoRoom panel
with exactly the same seeds, action bank, CEM budget, and frozen checkpoint as
the baseline. The readout was active and changed `420/480` supported bank
argmins (`87.5%`), but the task consequence was negligible:

| controller | success |
| --- | ---: |
| frozen LeWM baseline | `6/32` |
| goal-aware rank-affine readout | `7/32` |

There were zero baseline-only wins and one intervention-only win; exact paired
McNemar `p=1.0`, with delta `+1/32`. This is not a cross-environment gain.
It establishes a sharper failure mechanism: the runtime readout can alter
within-support action ranking while the learned ranking remains unrelated to
TwoRoom task success. The Reacher consequence therefore cannot be promoted to
a universal GDD-WM result.

The machine-readable paired audit is
`results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_capacity_latentrelative_v4_rankaffine_32/paired_summary.json`,
generated by
`le-wm-main/experiments/gdd_wm/summarize_tworoom_teacher_consequence.py`.

## Phase-conditioned fail-closed repair attempt

The TwoRoom failure suggests that a terminal-cost ranker cannot safely control
the door-crossing phase. A predeclared recovery arm was therefore added: RGB
phase detection forces the reactive waypoint controller during `door`, while
the GDD readout remains available during `goal`. This is a selective recovery
protocol, not a claim that the latent ranker is correct.

On the same 32 roots, the phase-conditioned hybrid reached `11/32`, compared
with `7/32` for the unrestricted goal-aware readout and `12/32` for the pure
RGB reactive controller. It recovers most of the loss but does not provide an
independent GDD gain. The artifact remains useful because it localizes the
failure to phase-conditioned task utility rather than generic runtime
activation.

The implementation is in
`le-wm-main/experiments/run_live_tworoom_planning.py` under
`--hybrid-door-phase-reactive`, with regression coverage in
`le-wm-main/tests/test_live_tworoom_planning.py`.

## Phase-conditioned GDD-WM teacher deployment audit

The train-only phase-conditioned teacher was exported with two runtime input
blocks that were wider than the original unconditioned contract: the action
summary became `12 + 2 + 24 = 38` dimensions, and the MLP input became
`192 + 38 = 230` dimensions. The runtime validator still assumed a fixed
12-dimensional action block and `latent + 12` MLP input. This was an
implementation/schema failure, not a planning result. The validator now
derives both widths from the exported artifact while retaining strict shape
checks. The repair is covered by the existing 50-test targeted suite and was
also validated by loading the actual phase-conditioned artifact.

The repaired teacher was then evaluated on the predeclared same-seed,
root-disjoint TwoRoom panel (32 roots, same frozen checkpoint, action bank,
CEM budget, horizon, and baseline). The result is:

| controller | success | paired consequence |
| --- | ---: | --- |
| frozen LeWM baseline | `6/32` | reference |
| phase-conditioned GDD readout | `6/32` | intervention-only `1`, baseline-only `1`, exact McNemar `p=1.0` |

The readout was active and changed `427/488` supported bank argmins (`87.5%`),
so this is not a no-op or deployment-fallback artifact. Nevertheless, the
ranking change produced no net task benefit. The machine-readable paired
summary is
`le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_phase_goal_latentrelative_32_retry2/paired_summary.json`.

This closes one engineering uncertainty and strengthens the scientific
boundary: RGB phase conditioning is sufficient to deploy the teacher, but
phase-conditioned terminal-cost ranking is still not sufficient for
door-crossing task utility. The current GDD-WM claim therefore remains
offline/calibration-positive and closed-loop-unverified across environments;
the next admissible repair is a train-only phase/waypoint-conditioned effect
or verified-local controller with an independent held-out consequence, not
further threshold tuning on this panel.

## Waypoint-conditioned teacher and action-only gate

The next repair separates two cases that the earlier gate conflated:

1. a latent residual that must beat a train-only action readout; and
2. a task-conditioned action readout that must beat the frozen LeWM score.

The second gate is now calibration-only. It compares the train-fitted
waypoint/phase action readout against the frozen model's recorded score and
enables the action readout only when its paired calibration regret interval is
strictly below zero. If only this gate passes, the runtime explicitly disables
the latent residual instead of silently mixing the two mechanisms.

The waypoint-conditioned artifact uses an outcome-free RGB direction to the
active door or goal waypoint, and a predeclared offline utility: midpoint
door-distance during the door phase, terminal goal-distance during the goal
phase. On calibration roots, the action-only gate passed with regret delta
`-45.56` and bootstrap 95% CI `[-61.46, -29.20]` versus frozen LeWM. The
latent residual gate did not pass.

The independent same-seed 32-root TwoRoom consequence was:

| controller | success | paired consequence |
| --- | ---: | --- |
| frozen LeWM baseline | `6/32` | reference |
| waypoint-conditioned action teacher | `7/32` | intervention-only `2`, baseline-only `1`, exact McNemar `p=1.0` |

The readout was active and changed `413/488` supported bank argmins (`84.6%`);
the residual correction was disabled. Thus the result is a genuine deployed
action-teacher intervention, but its held-out task improvement is not
statistically established. The artifact and paired summary are:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_phase_waypoint_target_goal_latentrelative_v2_actiongate/`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_waypoint_target_goal_latentrelative_32/paired_summary.json`

The current conclusion is therefore narrower and stronger: task-conditioned
waypoint features repair a teacher-gating omission and yield a positive point
estimate, but the remaining bottleneck is still reliable transfer from local
candidate ranking to door-crossing closed-loop success. No universal GDD-WM or
latent-causal-dynamics claim is promoted.

### Three-panel stability check

To distinguish a repeatable direction from one favorable seed panel, the same
frozen artifact and protocol were run on two additional independent 32-root
panels. The paired results were `6->7`, `5->9`, and `8->10` successes. Across
all 96 roots, frozen LeWM reached `19/96` and the waypoint teacher reached
`26/96`; intervention-only wins were `12` versus `5` baseline-only wins,
exact McNemar `p=0.1435`. All three panel summaries declare
`outcome_used_for_selection=false`.

This is a stable positive point estimate (`+7/96`) but not a statistically
established gain. The combined machine-readable audit is
`le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_waypoint_teacher_3panel_summary.json`.
The next repair target is therefore not more selector tuning: it is improving
the task-level transition/waypoint label and action support while preserving
the frozen gate, then validating on a separately held-out environment.

### Runtime-matched support panel and independent RGB consequence

The next negative result was traced to two separable causes. First, the
earlier support panel had only eight calibration roots, so the action-only
gate was directionally favorable but its interval crossed zero. Second, the
teacher audit requires history contexts to share the same root/action cells;
using different seeds correctly fails closed rather than silently matching
unpaired rows. We therefore rebuilt two history contexts on the same 64-root
panel with the exact runtime `vision_door16` bank (17 candidates per root),
then generated the waypoint labels offline from simulator replay. The
teacher fit used 32 train roots, 16 calibration roots, and 16 outcome roots;
the outcome roots were never used for selection.

The expanded calibration gate passed for the action-only teacher:

- action-only versus frozen regret delta: `-24.92`;
- paired bootstrap 95% CI: `[-41.13, -12.58]`;
- action gate: passed;
- latent residual gate: failed and was explicitly disabled;
- contexts: 2 history views, 2,176 records, 17 candidates/root;
- fit device: CUDA (`RTX 5060 Ti`, CUDA 13.0).

On an independent same-seed 32-episode RGB panel, using the same frozen
checkpoint, candidate bank, CEM budget, horizon, and seeds, the result was:

| controller | success | paired consequence |
| --- | ---: | --- |
| frozen LeWM, `vision_door16` | `11/32` | reference |
| support-matched waypoint action teacher | `14/32` | intervention-only `7`, baseline-only `4`, exact McNemar `p=0.549` |

The teacher changed `364/480` evaluated bank argmins (`75.8%`), so the
intervention was active. The point estimate is `+3/32` (`+9.375` percentage
points), but the paired test is not significant; this is an independent
positive trend, not an established closed-loop gain. The runtime explicitly
reported `residual_correction_enabled=false`, so this result cannot be
attributed to an unvalidated latent residual.

Artifacts:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_vision_door_teacher_actiongate_64shared/`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_vision_door_teacher_20262400_32/planning.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_vision_door_frozen_20262400_32/planning.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_live_vision_door_teacher_consequence_20262400_32.json`

The offline waypoint-label pass was also made deterministic and substantially
faster by limiting the renderer's tiny tensor compositing calls to one CPU
thread. This changes no teacher or deployment semantics. The remaining
repair boundary is now clear: calibration-supported action ranking transfers
directionally to RGB closed loop, but candidate ranking is still not a
reliable surrogate for multi-step door-crossing success. The next claim must
remain selective and environment-specific until the same gate and consequence
are reproduced on an unseen environment.

### Verified-local transfer repair: three independent held-out panels

The preceding teacher result isolated the remaining failure: the
support-matched action readout changed the local candidate preference, but a
multi-step door crossing could still fail when the learned action effect was
wrong. We therefore composed the frozen action teacher with the existing
online RGB effect verifier. At each re-observation, the verifier fits a
phase-specific 2-D action-to-marker map, requires rank-two support and a
held-out residual check, and only then uses the inverse map to repair a
rejected action. A failed support or residual check falls back to the
teacher/RGB gate; no held-out task outcome is consulted.

The repair was evaluated on three new, independent 32-root panels with the
same frozen teacher artifact, `vision_door16` support, CEM budget, horizon,
and paired seeds. The frozen baseline was run separately on each panel. The
combined result is:

| controller | success | paired consequence |
| --- | ---: | --- |
| frozen LeWM | `20/96` | reference |
| support-matched teacher + verified-local RGB repair | `45/96` | intervention-only `34`, baseline-only `9`, exact McNemar `p=1.70e-4` |

The per-panel transitions were `7->18`, `7->16`, and `6->11`. The repair arm
was active but selective: across panels, effect-map rank/residual gates
rejected unsupported local fits rather than forcing an inverse action. In the
first panel, for example, there were 167 verified-local interventions and
139 held-out-residual rejections. This makes the result a closed-loop
consequence of decision-space verification, not a post-hoc relabeling of the
teacher. Because this repair arm was introduced after the earlier negative
teacher result, the p-value is reported as held-out evidence for this frozen
repair protocol, not as a preregistered claim over an unlimited method search.

The machine-readable artifacts are:

- `le-wm-main/experiments/gdd_wm/summarize_tworoom_repair_consequence.py`
- `le-wm-main/experiments/gdd_wm/summarize_tworoom_repair_panels.py`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_verified_local_repair_3panel_summary.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_verified_local_vs_frozen_20262900_32.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_verified_local_vs_frozen_20263000_32.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_verified_local_vs_frozen_20263100_32.json`

The updated methodological claim is selective rather than universal:

> A task-conditioned teacher becomes useful for frozen visual planning when
> candidate support is matched and the selected action is verified against
> an online, phase-local effect model; unsupported effect estimates must
> fail closed.

This repairs the previous closed-loop negative result on TwoRoom. It does not
yet establish universal transfer, a universal learned-world-model theorem,
or a cross-environment guarantee; those still require the same frozen gate
and independent consequence on an unseen environment.

### PointMaze transfer-safety replication and ceiling boundary

To test whether the repair mechanism was specific to TwoRoom, the existing
history-4 RGB effect policy was frozen on calibration episodes `1800--1815`
and evaluated on a new episode-disjoint PointMaze panel `1890--1899` using
CUDA. The policy received RGB-derived observations only; simulator state was
used for stepping and evaluation, not policy input.

On this panel, visual dynamics and the verified-local effect policy both
reached `10/10` success. The paired consequence is therefore `0/10` with
McNemar `p=1.0`, while the verified-local arm had mean final distance
`0.160` versus `0.120` for visual dynamics. This is not a failed
implementation: it is a ceiling effect. PointMaze already saturates the
baseline, so it validates transfer safety of the gate but cannot identify an
incremental planning gain.

The report and paired audit are:

- `le-wm-main/results/gdd_wm_pointmaze_independent_heldout_20261019_cuda_history4_1890_1899.json`
- `le-wm-main/results/gdd_wm_pointmaze_independent_heldout_20261019_cuda_history4_1890_1899_paired.json`
- `le-wm-main/experiments/gdd_wm/summarize_pointmaze_paired_consequence.py`

This separates two claims that had previously been conflated: the
verified-local mechanism transfers beyond TwoRoom, but a saturated easy
environment cannot establish additional task success. The decisive positive
incremental evidence remains the three-panel TwoRoom result; PointMaze is an
independent safety/portability check, not a second positive-gain claim.

### Cube phase-entry repair: independent positive consequence

The first fresh Cube panel exposed why the same verified-local controller was
mostly inert under the original fixed prelude: the controller could fit an
RGB local effect map, but the pre-contact descent/closure phase did not put
many test roots into a useful action-effect regime. On the original 16-root
panel, verified-local changed actions but had 13/16 tied outcomes and a mean
final-distance delta of `-0.00423 m`; this is insufficient evidence of a
stable consequence.

The repair was to add a fixed, RGB-only approach phase before the unchanged
descent/closure prelude. The approach duration and gain were frozen before the
new test roots; calibration used 40 roots and the planning panel used 16 new
paired roots. No simulator state or test outcome entered the approach rule,
local fit, gate, or candidate selection.

On the independent CUDA panel:

| arm | mean final-distance delta vs static | wins | losses | ties | bootstrap 95% CI | exact sign p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| verified RGB local effect | `-0.03053 m` | 9 | 0 | 7 | `[-0.05193,-0.01299]` | `0.00390625` |
| verified local + frozen geometry fusion | `-0.03019 m` | 9 | 0 | 7 | `[-0.05171,-0.01226]` | `0.00390625` |
| agreement-gated verified local | `-0.03053 m` | 9 | 0 | 7 | `[-0.05158,-0.01306]` | `0.00390625` |

The original temporal predictor and support-only gate remained non-positive on
the same panel (`+0.00064 m`, 0/1 non-tie improvements), so the gain is not a
generic temporal-memory or threshold effect. The result supports a narrower
mechanism: a task-conditioned RGB approach phase makes local effect
verification actionable, after which fail-closed local ranking can improve
the frozen planner without privileged runtime state.

Artifacts:

- `le-wm-main/results/gdd_wm_cube_verified_local_20261019_cuda_fresh16/closed_loop.json`
- `le-wm-main/results/gdd_wm_cube_verified_local_approach_20261019_cuda_fresh16/closed_loop.json`
- `le-wm-main/results/gdd_wm_cube_verified_local_approach_20261019_cuda_fresh16/aggregate.json`

The aggregation code now includes explicit `approach_steps` and
`approach_gain` in its protocol signature and rejects pooling panels with
different phase-entry protocols. This prevents the positive approach result
from being silently mixed with the original negative panel. The result is a
new cross-environment repair consequence, not evidence for universal GDD-WM
transfer, causal latent dynamics, or universal world-model claims.

### RGB-successor-only preference transfer on Cube

To separate the phase-entry repair from privileged task labels, a second Cube
panel fit the macro preference readout on `rgb_goal` targets: calibration
targets were realized RGB successor goal errors, not simulator task costs.
The panel used 40 calibration roots and 16 new paired test roots on CUDA, with
the same root-disjoint protocol and a frozen approach/contact bank.

Relative to the fullscale reclose baseline, the low-gain effect-adaptive arm
obtained an intervention-minus-baseline final-distance delta of `-0.03441 m`
with paired bootstrap 95% CI `[-0.06709,-0.00590]`; it improved 10 roots and
degraded 6. The exact two-sided sign test was `p=0.4545`, so the interval and
point estimate should not be described as a statistically established
universal gain. The success counts were `2/16` versus `0/16`, also too small
for a standalone success-rate claim. The stricter effect-stop arm was weaker:
delta `-0.01708 m`, CI `[-0.04610,0.00487]`.

This result narrows the causal interpretation: RGB realized-effect
preference plus low-gain contact adaptation can improve distance on a fresh
panel, but fail-closed stopping does not automatically improve the endpoint,
and the evidence is still not a frozen LeWM/D-JEPA teacher consequence. The
machine-readable adjudication is:

- `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_20261019_cuda_fresh16/panel.json`
- `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_20261019_cuda_fresh16/paired_summary.json`
- `le-wm-main/experiments/summarize_cube_rgb_macro_panel.py`

### Cube approach-horizon repair and residual task boundary

The RGB-successor-only panel was repeated with a predeclared 20-step RGB
approach on 16 new roots. The adaptive arm improved final distance on 12/16
roots and degraded 4/16, with intervention-minus-baseline mean `-0.03441 m`
and bootstrap 95% CI `[-0.05458,-0.01598]`; the exact sign test was
`p=0.0768`. Contact observation increased to `0.8125` overall. The effect is
therefore directionally reproducible, but the endpoint success rate was only
`1/16` and the test is not a success-rate result.

Task-stratified contact rates show the mechanism and its boundary: task 4
increased from `1/3` to `2/3` contact roots, while task 5 remained `1/3`.
The longer approach fixes part of the pre-contact support problem, but does
not solve the task-5 geometry/contact regime. This is useful negative
evidence: a longer history/approach phase is not a universal solution, and
remaining task failure must not be attributed to the teacher ranker alone.

Artifacts:

- `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_approach20_20261019_cuda_fresh16/panel.json`
- `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_approach20_20261019_cuda_fresh16/paired_summary.json`

## GDD-WM teacher-to-effect closed-loop repair on Cube task 5 — 2026-10-19

The task-5 approach sweep first isolated a protocol-level source of the
earlier negative result.  With the fixed 15-step descent and no RGB approach,
the front-view contact proxy had mean distance `0.55693` and `0/16` roots inside
the predeclared `0.05` support threshold.  A calibration-only RGB routing sweep
selected `approach_steps=20, approach_gain=1.0`: front distance mean `0.02601`
and `15/16` roots inside support.  The selection used only calibration roots
and the RGB contact proxy; test success was not read.

The frozen temporal ranker and the verified-local recovery were then evaluated
on 16 new task-5 roots.  The ranker alone did not improve the paired distance:
mean delta versus the static arm was `+0.00896 m`, bootstrap 95% CI
`[-0.00482, 0.02252]`, with `5/16` wins and `9/16` losses among non-ties.  This
is a teacher-to-action-effect failure, not evidence that the GDD interface is
planning-sufficient.

The same frozen candidate bank was augmented with a runtime RGB effect map.  It
requires at least four observed rows, rank-two support, and a finite condition
number; otherwise it falls back to the fixed four-direction carrier.  It uses
current/previous RGB, the rendered goal, previous action, and finite phase
state, while simulator state is restricted to paired reset and final scoring.
The resulting held-out consequence was:

| arm | mean final-distance delta vs static | bootstrap 95% CI | wins / losses / ties |
| --- | ---: | ---: | ---: |
| frozen temporal ranker | `+0.00896 m` | `[-0.00482, 0.02252]` | `5 / 9 / 2` |
| verified RGB effect | `-0.01742 m` | `[-0.03025, -0.00601]` | `7 / 0 / 9` |

The exact two-sided sign-test p-value for the verified arm after removing ties
is `0.015625`.  The artifact is
`le-wm-main/results/gdd_wm_cube_task5_teacher_closed_loop_20261019_cuda_fresh16/closed_loop.json`,
with the frozen-protocol aggregation in `aggregate.json`.  This is a scoped
task-5 distance-level result: it supports the recovery mechanism

`frozen teacher candidate proposal -> RGB effect verification -> fail-closed fallback`

but does not establish D-JEPA teacher correctness, task-success improvement,
universal transfer, or a universal GDD-WM certificate.  The central negative
result remains scientifically important: decision distillation becomes useful
only after the teacher proposal is checked against the realized local action
effect.

## PointMaze GDD projection student: verified-local repair audit — 2026-10-19

The D2 projection student was connected to the live PointMaze runtime as
`gdd_wm_rgb_verified_local`.  The new arm uses the frozen projection student
to propose candidates, executes four fixed RGB information probes, fits an
observed affine action-effect chart, and releases the local proposal only
when the chart has rank-three affine support and its predicted local regret
against the calibration-only visual-dynamics action is at most `0.10`.
Otherwise it falls back to the visual-dynamics candidate.  The probes are
waypoint-aligned with the RGB maze planner and use a fixed magnitude `0.25`.
No teacher rank, candidate ID, simulator state, or held-out outcome is passed
to the runtime student.

The first run exposed two implementation errors and was discarded as a
scientific artifact: the new method was missing from the dispatch branch, and
the probes were aimed at the global goal instead of the current RGB maze
waypoint.  After both repairs, the root-disjoint 20-root replication panel
reported:

| controller | success | mean final distance | mean progress |
| --- | ---: | ---: | ---: |
| visual dynamics | `20/20` | `0.143` | `1.526` |
| frozen GDD projection | `3/20` | `2.300` | `-0.631` |
| GDD projection + verified local | `3/20` | `1.347` | `0.322` |

Thus observed-effect verification repairs a substantial part of the frozen
GDD student's distance failure (`2.300 -> 1.347`) and eliminates almost all
local gate fallback (`1.1%` of decisions), but it does not beat the saturated
visual-dynamics controller on success.  This is a useful mechanism result, not
an independent GDD planning gain.  It also shows that the remaining failure
is not only action-effect ranking: the four information probes consume a
large part of the motion budget, and PointMaze visual dynamics is already at
the task ceiling on this panel.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_gdd_verified_local_20261019_cuda_v8_waypoint_consistency.json`
- `le-wm-main/experiments/run_live_pointmaze_planning.py`

## GDD full predicted-dynamics runtime contract — 2026-10-20

The prior capacity artifacts contained `bilinear_full_*` coefficients, but the
TwoRoom runtime had no explicit deployment switch.  As a result, an offline
full-dynamics audit could be reported while the live controller still used the
artifact's default action-only path.  The runtime now exposes the explicit
`--capacity-bilinear-mode {artifact,action_only,full_predicted_dynamics}`
override.  It only selects coefficients already exported by the train-only
artifact, records the selected mode in the protocol, and rejects full mode for
goal- or phase-conditioned artifacts whose exported feature chart is
incompatible.

The corrected runtime was first validated with 45 focused tests and a host
CUDA preflight (RTX 5060 Ti, PyTorch `2.12.0+cu130`).  A fresh paired
64-seed TwoRoom panel then used the same frozen checkpoint, legacy candidate
bank, CEM settings, and seeds as the raw frozen control:

| controller | success | vs raw frozen LeWM (wins/losses/ties) |
| --- | ---: | ---: |
| raw frozen LeWM | `18/64` | — |
| explicit action-only bilinear | `8/64` | `0/10/54`, exact McNemar `p=0.00195` |
| explicit full predicted-dynamics bilinear | `9/64` | `0/9/55`, exact McNemar `p=0.00391` |

The runtime diagnostics confirm that full mode was actually active: gate
enabled, selected mode `bilinear_full_predicted_dynamics`, `968` evaluated
candidate rows, and `80.3%` candidate-bank argmin changes.  Nevertheless it
reduced success relative to raw frozen LeWM.  The paired success deltas were
`-0.15625` for action-only and `-0.140625` for full predicted dynamics; paired
bootstrap 95% intervals were `[-0.25,-0.078125]` and
`[-0.234375,-0.0625]`, respectively.  The action-only control was also
negative, so the immediate cause is not only the predicted-delta interaction;
the train-only capacity readout's candidate ranking is not calibrated to the
closed-loop TwoRoom utility.  This resolves the implementation ambiguity but
does not produce a positive teacher consequence.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_full_bilinear_teacher_fresh64_20261020_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_action_bilinear_teacher_fresh64_20261020_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_raw_frozen_control_fresh64_20261020_cuda/planning.json`
- `le-wm-main/experiments/capacity_readout_runtime.py`

The machine-readable four-panel audit is now materialized at
`le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_verified_local_repair_4panel_summary.json`.
It records the panel-level runtime diagnostics rather than only the success
counts: 7,643 verification-gate calls, 1,333 fail-closed fallbacks (17.44%),
778 effect-policy validation fallbacks, and 1,625 released effect-policy
decisions.  The weighted verification cosine is `0.314`; its panel P10 values
remain negative, so this protocol is not a directional action certificate.
The positive consequence therefore belongs to the complete
`teacher proposal -> RGB verification -> local effect map -> fallback`
controller, not to the frozen teacher in isolation.

The four panels are the pre-existing 20263200, 20263300, 20263400, and
20263500 root blocks.  They were not selected by the new outcome summary; the
summary script rejects mismatched seed arrays before computing paired counts.
This makes the artifact suitable as a reproducible TwoRoom follow-up, while
the sequential panel expansion and the negative verification-direction tail
still prevent a universal GDD-WM or causal-latent claim.
- `le-wm-main/experiments/run_live_tworoom_planning.py`

The result is therefore a valid negative boundary: offline teacher capacity,
runtime activation, and candidate-rank changes are insufficient for an
independent planning gain.  The formal GDD claim remains unverified; the next
repair must target task-conditioned closed-loop calibration rather than
silently changing the runtime mode.

## TwoRoom planner-ceiling and candidate-support audit — 2026-10-20

The review's planner-confound test was run before further teacher tuning.  A
privileged evaluator-side oracle used the same horizon `5` and action block
`5`, but generated 289 root-local candidates and scored their real successors.
On 64 fresh roots, only `35/64` roots (`54.69%`) had any successful candidate;
`29/64` had no successful candidate in this horizon/action family.  This is a
hard reachability ceiling and cannot be repaired by GDD geometry.

To test whether the smaller RGB bank was the cause, the frozen LeWM was run on
the same 64 roots with two predeclared banks:

| bank | frozen LeWM | RGB reactive | verified hybrid |
| --- | ---: | ---: | ---: |
| `vision_door16` | `24/64` | `35/64` | `41/64` |
| `angular16` | `25/64` | `35/64` | `39/64` |

The paired frozen-LeWM difference is only `+1` root (exact paired McNemar
`p=1.0`); the reactive arms are identical, and the hybrid difference is `-2`
roots (`p=0.5`).  Thus candidate angular coverage is not the dominant cause of
the verified-local gain or the remaining teacher failure on this panel.  The
negative result is now separated into (i) a genuine planner/reachability
ceiling on some roots and (ii) within-support action-effect uncertainty on
reachable roots.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_goal_directed_oracle_20261020_fresh64/report.json`
- `le-wm-main/results/gdd_wm_tworoom_visiondoor16_control_20261020_cuda_fresh64/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_angular16_coverage_20261020_cuda_fresh64/planning.json`

## Goal-conditioned bilinear chart and verified-local recovery — 2026-10-20

The preceding negative panel used a no-goal bilinear artifact for a
goal-conditioned TwoRoom task.  That was a real feature-contract limitation:
the offline residual saw goal-relative latent features, while the deployed
bilinear teacher did not.  The capacity audit now exports a train-only PCA
chart for `goal - source`, its goal/action interaction, and the
goal-relative/predicted-delta interaction.  Runtime validates the new chart
and exposes it through the same explicit bilinear mode override.

On the matched 64-root TwoRoom panel with the training-consistent
`vision_door16` support:

| controller | success |
| --- | ---: |
| raw frozen LeWM | `17/64` |
| goal-conditioned action-only bilinear | `13/64` |
| goal-conditioned full predicted-dynamics bilinear | `13/64` |
| goal-conditioned full teacher + online RGB effect verification | `21/64` |

The full teacher was genuinely active (`968` candidate rows and `81.7%`
bank-argmin change rate), but goal conditioning alone did not repair the
closed loop.  The online effect verifier increased success by checking the
realized RGB displacement with a phase-local action-effect map and falling
back when its held-out residual gate failed.

Two additional independent 32-root panels used the identical frozen artifact,
support, CEM budget, and verification protocol.  The verified-local arm gave
`10/32` versus raw `9/32`, and `11/32` versus raw `9/32`.  A further
independent 64-root panel gave `30/64` versus raw `20/64`.  Across all four
panels (`192` roots), the paired consequence was:

- intervention-only wins: `33`;
- baseline-only wins: `16`;
- ties: `143`;
- success-rate delta: `+0.0885`;
- paired bootstrap 95% CI: `[+0.0156,+0.1615]`;
- exact McNemar `p=0.0213`.

This is now a statistically supported positive consequence for the frozen
TwoRoom protocol, although the expanded panel was added sequentially and
should be treated as confirmatory follow-up rather than a preregistered claim.
The mechanism is localized more precisely: goal-relative latent features are
necessary for the task chart, while online effect verification is what
converts candidate preference into a closed-loop improvement.  The full
predicted-dynamics term itself remains calibration-gated and was not
independently validated (`full-vs-no-dynamics` calibration interval crosses
zero).  The result therefore supports a scoped GDD-WM recovery protocol, not
a universal teacher, universal world model, causal latent, or commutator
certificate.

Artifacts:

- `le-wm-main/results/gdd_wm_goal_bilinear_teacher_20261020_cuda/capacity_audit.json`
- `le-wm-main/results/gdd_wm_tworoom_goal_bilinear_full_verified_fresh64_20261020_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_goal_bilinear_full_verified_20263300_32_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_goal_bilinear_full_verified_20263400_32_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_goal_bilinear_full_verified_20263500_64_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_goal_bilinear_raw_20263500_64_cuda/planning.json`
- `le-wm-main/experiments/gdd_wm/audit_train_only_teacher_capacity.py`
- `le-wm-main/experiments/capacity_readout_runtime.py`

## GDD action-support contract audit — 2026-10-20

The frozen package metadata declared a 16-candidate bank, but the actual
repeated-context package used for the earlier D2 projection contained only 11
candidate records per root/context.  The live runtime scored 16 candidates.
The runner now exposes `--gdd-candidate-count` and records it in the protocol,
so this mismatch cannot remain implicit.

As a matched control, the same fresh CUDA PointMaze panel was rerun with the
training-consistent 11-candidate prefix and `h=4` scoring:

| GDD candidate support | success | mean final distance |
| --- | ---: | ---: |
| runtime 16 candidates | `3/20` | `2.4277` |
| training-consistent 11 candidates | `3/20` | `1.6679` |
| visual dynamics | `20/20` | `0.1631` |

The support repair improves the continuous error relative to the 16-candidate
`h=4` run but does not improve success or establish a closed-loop GDD gain.
The full 16-candidate train-only CUDA capacity audit also failed to show a
reliable predicted-dynamics increment over action-only ranking on outcome
roots (`mean regret delta=+0.0481`, 95% CI `[-0.0745,+0.1687]`).  Thus the
remaining negative result is not explained by action support alone: the
teacher's decision preference is still not sufficiently stable for an
independent student planning claim.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_support11_20261020_cuda_fresh20.json`
- `le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261020_full16_cuda/capacity_audit/capacity_audit.json`
- `le-wm-main/experiments/run_live_pointmaze_planning.py`

The implementation is retained as a fail-closed diagnostic/recovery arm.  It
does not authorize a claim of D-JEPA teacher correctness, universal decision
distillation, or a cross-environment planning certificate.

### Action-support gate follow-up

The GDD paired action family uses first-pulse magnitudes up to `1.0`, whereas
the verified-local information probes use fixed magnitude `0.25`.  This is an
action-support mismatch, so the local chart was changed to release only
candidate actions within `1.5 * 0.25` of the probe support.  Fallback visual
dynamics actions are no longer written into the local probe chart.

On the same 20-root paired panel, the support gate changed the verified-local
arm from `3/20` to `4/20` success and mean final distance from `1.347` to
`1.343`; the paired distance delta was only `-0.00358` m with `10/20` roots
improving and `10/20` worsening.  The standalone GDD arm remained `3/20` and
`2.300` m.  Therefore the support gate fixes an extrapolation pathway and
slightly improves the point estimate, but it is not yet an independent
planning gain.

Artifact:
`le-wm-main/results/gdd_wm_pointmaze_gdd_verified_local_20261019_cuda_v9_support_gate.json`.

### Disagreement-triggered information budget — fresh CUDA panel

The preceding PointMaze repair paid for four RGB probes even when the frozen
GDD proposal agreed with the calibration-only visual-dynamics proposal.  This
confounded local-effect repair with an avoidable execution cost.  The new
controller, `gdd_wm_rgb_disagreement_verified_local`, performs a dry proposal
comparison using only current RGB and frozen calibration artifacts, and runs
the four fixed magnitude-`0.25` probes only when their direction cosine is
below the predeclared `0.80` threshold.  A skipped probe is not counted as an
executed action in the artifact budget.

On a fresh root-disjoint CUDA panel (`calibration=1800--1815`,
`test=1868--1887`), the result was:

| controller | success | mean final distance | mean executed probes |
| --- | ---: | ---: | ---: |
| visual dynamics | `20/20` | `0.1631` | `0.0` |
| frozen GDD projection | `6/20` | `1.7400` | `0.0` |
| disagreement-triggered verified-local | `11/20` | `0.8136` | `3.0` |

The new arm triggered probes on `15/20` roots (`75%`), improved final distance
on `16/20` roots versus frozen GDD, and changed success on `6` roots in its
favor versus `1` in the opposite direction.  It still lost to the saturated
visual-dynamics baseline (`0/9` success transitions in its favor and `9` in
the opposite direction).  Thus the intervention repairs the previously
identified GDD failure and reduces unnecessary probing, but does not create an
independent PointMaze gain.  The remaining negative result is now localized:
the learned student can be rescued relative to its own proposal, while the
RGB dynamics controller already solves this panel and leaves no headroom for
an incremental teacher gain.

The implementation and artifact are:

- `le-wm-main/experiments/run_live_pointmaze_planning.py`
- `le-wm-main/results/gdd_wm_pointmaze_gdd_disagreement_verified_local_20261019_cuda_fresh20.json`

This result supports a scoped rule—`probe only on frozen-proposal
disagreement, otherwise fail closed`—and does not authorize a universal GDD
teacher, universal transfer, or causal-latent claim.  The corrected budget
counter excludes skipped-probe trace records.

### Post-action verification and RGB path support — fresh CUDA re-audit

The disagreement arm was still releasing a local action after a single-step
chart fit without checking the realized effect of that released action.  We
added two fixed, RGB-only safeguards: (i) after every released local pulse,
compare predicted and observed marker displacement and quarantine the
episode-local chart when the residual exceeds `0.25`; (ii) reject a local
predicted endpoint when its sampled RGB-space segment leaves the
calibration-inferred free-cell support.  A rejected transition is not written
back into the chart.

On the same `1800--1815` calibration and `1868--1887` test roots, the complete
path-gated panel reported:

| controller | success | mean final distance | post-action verification | chart quarantine |
| --- | ---: | ---: | ---: | ---: |
| frozen GDD projection | `6/20` | `1.7400` | n/a | n/a |
| disagreement + post-verification + path gate | `13/20` | `0.6906` | `98.35%` | `40%` of roots |
| visual dynamics | `20/20` | `0.1631` | n/a | n/a |

The repaired arm improved final distance on `15/20` roots versus frozen GDD;
the success discordance was `8` intervention-only versus `1` baseline-only.
The occupancy gate rejected all local candidates on `39` decision calls, so it
is active rather than documentary.  The intervention remains below the
visual-dynamics ceiling (`0/7` success transitions in its favor versus `7`
baseline-only), but the sequence of failures is now narrower: unverified
single-step errors and RGB-inferred wall-crossing are both fail-closed, while
the remaining errors are multi-step route/task-utility failures.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_gdd_pathgated_20261019_cuda_fresh20_correct.json`
- `le-wm-main/experiments/run_live_pointmaze_planning.py`

### Route-phase quarantine — multi-step failure repair

The path-gated arm still reused an episode-local chart after the RGB planner
changed its active waypoint.  We therefore added a phase gate: a waypoint
change quarantines the old chart immediately, executes that transition block
with the RGB visual-dynamics fallback, and prevents further local releases in
the new phase.  This uses only the current RGB waypoint and the calibration
map; it does not inspect simulator state or held-out outcomes.

On the same frozen, root-disjoint CUDA panel, the complete phase-gated arm
reported:

| controller | success | mean final distance |
| --- | ---: | ---: |
| frozen GDD projection | `6/20` | `1.7400` |
| post-verified + path + phase quarantine | `15/20` | `0.4863` |
| visual dynamics | `20/20` | `0.1631` |

Relative to frozen GDD, distance improved on `16/20` roots and success
discordance was `10` intervention-only versus `1` baseline-only.  The phase
gate was invoked on `26` transition blocks across the panel.  This is the
strongest PointMaze repair so far and directly supports the mechanism claim
that a local action-effect chart is phase-conditional.  It still does not
beat the saturated visual-dynamics baseline: relative to that baseline there
were `0` intervention-only and `5` baseline-only success transitions.  The
remaining gap is therefore not merely single-step effect identifiability; it
includes probe cost and residual multi-step task utility.

Artifact:
`le-wm-main/results/gdd_wm_pointmaze_gdd_phasegated_20261019_cuda_fresh20.json`.

## GDD projection horizon-contract audit — 2026-10-20

The frozen projection was fitted from candidate records declared with
`predicted_horizon=4`, while the live PointMaze scorer previously averaged
the projection energy from `h=1,2,4`.  This was an implementation-level
train/deployment mismatch.  The runner now exposes an explicit
`--gdd-score-horizons` contract, defaults to the distillation horizon `h=4`,
validates the selected horizons against the frozen adapter, and records them
in the protocol artifact.  The old multi-horizon behavior remains available
as an explicit control (`--gdd-score-horizons 1 2 4`).

On the same fresh CUDA panel (`calibration=1800--1815`,
`test=1868--1887`, dataset-matched RGB renderer), the contract repair did not
create an independent planning gain:

| GDD scoring policy | success | mean final distance |
| --- | ---: | ---: |
| explicit `h=4` | `3/20` | `2.4277` |
| legacy `h=1,2,4` control | `3/20` | `1.6193` |
| visual dynamics | `20/20` | `0.1631` |

The repair is retained because it removes a real protocol violation, but the
matched result localizes the remaining failure to teacher/student geometry
and action utility rather than horizon averaging alone.  In particular,
changing the scoring horizon changes the continuous error but not the success
rate, and neither frozen student beats the RGB dynamics baseline.  This is a
negative teacher-to-planning propagation result, not evidence for a positive
GDD teacher claim.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_horizon4_20261020_cuda_fresh20.json`
- `le-wm-main/results/gdd_wm_pointmaze_horizon124_20261020_cuda_fresh20.json`
- `le-wm-main/experiments/run_live_pointmaze_planning.py`

## Exact runtime-chart policy and active-probe audit — 2026-10-20

The formal exact-chart runtime policy was fit on train roots only and then
evaluated on unopened roots `1868--1887` with a real CUDA allocation on the
RTX 5060 Ti.  The four-arm report is:

| controller | success | mean final distance | interpretation |
| --- | ---: | ---: | --- |
| visual dynamics | `20/20` | `0.1485` | RGB baseline ceiling on this panel |
| frozen effect adapter | `2/20` | `2.4041` | frozen teacher/action-effect ranking fails |
| online residual | `2/20` | `2.4041` | no residual was released; no usable verified samples |
| HTSC fallback | `20/20` | `0.1485` | exact RGB fallback, not GDD gain |

The frozen policy's calibration direction gate is invalid (`mean cosine
`-0.028`, P10 `-0.982`) and its calibrated regret bound is `0.4322`, above the
release tolerance.  Consequently the online residual arm correctly abstained
rather than silently claiming adaptation.  This is a genuine closed-loop
negative result: exact action-chart matching does not make the frozen teacher
action-identifiable.

The predeclared RGB-only active-probe/local-Jacobian arm was then run on the
same roots.  It executed four probes per root and reached `20/20` success with
mean final distance `0.1688`; the matched visual baseline remained `20/20`
with `0.1485`.  The local chart was trusted, but its proposed actions failed
the visual-dynamics consistency gate on `327/352` decisions.  Thus probing
recovers a usable local chart relative to the frozen teacher but does not beat
the already-saturated RGB controller and incurs probe cost.  The result
supports a scoped mechanism claim—action effects are local and phase
conditional—but not an independent GDD teacher gain.

The paired consequence is `0/20` intervention-only versus `0/20` baseline-only
(exact McNemar `p=1.0`); mean final-distance delta (intervention minus baseline)
is `+0.02034`.  The frozen adapter comparison is materially negative:
`0/20` intervention-only versus `18/20` baseline-only (exact McNemar
`p=7.63e-06`), with distance delta `+2.25560`.

Two runtime logging bugs exposed by this experiment were repaired and covered
by the PointMaze regression suite: the RGB-only local-Jacobian probe no longer
calls a missing frozen policy, and probe traces report `null` verification
error when no teacher prediction exists.  Targeted tests now pass `32` cases.

Artifacts:

- `le-wm-main/results/gdd_wm_online_local_effect_20261019_cuda_fresh20/planning_report.json`
- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_jacobian_20261020_cuda_fresh20/planning_report.json`
- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_jacobian_20261020_cuda_fresh20/paired_local_vs_visual.json`
- `le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_cuda.sh`

## Reacher cross-seed negative-transfer decomposition — 2026-10-20

To test whether the Reacher negative consequence came specifically from the
predicted-dynamics interaction terms, the frozen capacity artifact was exposed
through an explicit `--capacity-bilinear-mode` ablation.  No coefficient was
refit and the 128-root seed panel was fixed before evaluation.  Relative to
the same frozen-LeWM baseline (`64/128`, mean final distance `0.2140`):

| frozen readout | success | mean final distance | paired distance delta |
| --- | ---: | ---: | ---: |
| full predicted dynamics | `60/128` | `0.2160` | `+0.00196` |
| action-only | `59/128` | `0.2299` | `+0.01593` |
| full + fixed 0.5 base-cost blend | `59/128` | `0.2164` | `+0.00242` |

The full-vs-baseline success CI is `[-0.09375, 0.03125]`; action-only and the
0.5 hybrid also fail to improve success.  Therefore the predicted-dynamics
interaction is useful relative to action-only on this panel, but the complete
readout still does not transfer beyond the frozen base controller.  The
remaining cause is best described as cross-seed action-ranking shift, not an
implementation-only bias-order or base-cost-replacement bug.

The runtime now records and validates the selected bilinear component and the
optional fixed base-cost blend.  These are explicit ablations; they do not
authorize selecting a mode from the held-out panel.

A fixed CEM timing repair was also tested: keep the candidate bank active for
all eight CEM iterations and enable the bilinear readout only at iteration 7.
It yielded `60/128` successes and mean final distance `0.3133`; the paired
distance delta was `+0.09930` with 95% CI `[+0.02845,+0.18300]`.  This is worse
than applying the readout during the first seven iterations, so the original
negative transfer is not explained by a bank/final-stage scheduling mismatch.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_independent_baseline_128_cuda/planning.json`
- `le-wm-main/results/gdd_wm_reacher_independent_capacity_128_cuda/paired_comparison.json`
- `le-wm-main/results/gdd_wm_reacher_independent_action_only_128_cuda/paired_comparison.json`
- `le-wm-main/results/gdd_wm_reacher_independent_hybrid05_128_cuda/paired_comparison.json`
- `le-wm-main/results/gdd_wm_reacher_independent_finalstage_128_cuda/paired_comparison.json`

## TwoRoom phase-fix matched teacher consequence — 2026-10-20

The TwoRoom runtime previously allowed the verified-local effect-policy branch
to overwrite the declared door-phase RGB fallback.  The runner now carries an
explicit `phase_effect_release_allowed` condition, blocks effect-policy
release during the door phase, and records `effect_policy_phase_block_count`.
The regression suite includes a test that prevents this gate from being
overridden.

After the repair, a fresh 64-root CUDA intervention was matched to a fresh
64-root frozen baseline (same seeds, candidate bank, horizon, action block,
and planning budget).  The results are:

| controller | phase-fix teacher panel | matched frozen baseline |
| --- | ---: | ---: |
| frozen LeWM | `16/64` | `14/64` |
| verified-local hybrid | `28/64` | `27/64` |
| RGB reactive | `31/64` | `31/64` |

The frozen-teacher comparison has 3 teacher-only successes and 1
baseline-only success (exact McNemar `p=0.625`).  The verified-local hybrid
has 15 wins, 12 losses and 37 ties relative to the same-panel frozen LeWM
arm; this is a recovery/control diagnostic, not evidence of an independent
teacher gain.  The hybrid's effect-policy runtime recorded 381 releases,
933 door-phase blocks, 302 held-out validation fallbacks, and 1,392 total
fallbacks over 2,476 gate calls.  Thus the phase-fix removed a real runtime
confound, but the held-out closed-loop consequence remains statistically
inconclusive and the RGB reactive controller remains the ceiling on this
panel.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_verified_local_phasefix_fresh64_20261020_cuda/planning.json`
- `le-wm-main/results/gdd_wm_tworoom_verified_local_phasefix_fresh64_20261020_cuda_baseline/planning.json`
- `le-wm-main/experiments/run_live_tworoom_planning.py`
- `le-wm-main/tests/test_live_tworoom_planning.py`

The current scientific diagnosis is therefore narrower: the original
negative result contained a genuine gate-ordering bug, which is fixed; after
the fix, the remaining limitation is action-effect ranking/support under
phase transitions, not an implementation claim that can be repaired by
turning the teacher on more often.  GDD-WM teacher independent gain remains
unverified on this TwoRoom panel.

## Fresh causal future/task teacher retrain — 2026-10-20

To separate an old artifact from the teacher mechanism itself, the
WA-JEPA-inspired causal RGB future/task-effect adapter was retrained from
scratch on the canonical history-4 candidate bank with 250 fixed epochs on
CUDA.  The fit used train roots only; calibration/outcome roots were not used
for parameter or epoch selection, and no future observation or privileged
state was an input.  This is an execution-supervised control teacher, not a
validated D-JEPA teacher.

For the main asymmetric joint head, task-effect RMSE was `0.27352`, RGB
residual RMSE was `0.18885`, and the held-out candidate ranking regret was
`0.02342` versus random regret `0.02404` (mean paired regret gain
`+0.00062`).  The selected-success rate was `0/16`.  The best offline ranking
variant in this run (`asym_rank`) reached regret `0.02077` and paired gain
`+0.00327`, but this is a small offline control result with no closed-loop
consequence and therefore does not open a teacher claim.

The retrain localizes the remaining problem more sharply: joint future
prediction and physical-effect fit are measurable, but the candidate spread
is small and the resulting preference signal does not robustly identify the
successful action.  Consequently, the next teacher gate must require
root-disjoint preference stability plus an independent closed-loop consequence
before any projection distillation is accepted.

Artifact:

- `le-wm-main/results/gdd_wm_teacher_retrain_canonical_hybrid_20261020_cuda`

Its protocol explicitly records `device=cuda`, train-root-only fitting,
`runtime_future_observation_input=false`, `runtime_privileged_state=false`,
and `closed_loop_online_planning=false`.

## Reacher terminal-stability gate sweep — 2026-10-20

The Reacher negative-transfer decomposition was followed by a predeclared
gate-strength sweep.  The hypothesis was that the previous teacher changed
too many bank argmins and occasionally displaced a terminally stable frozen
candidate.  The intervention was therefore restricted to the lower `25%` of
the frozen base-cost bank; a stricter `10%` arm was retained as a diagnostic
control.  Both were run with CUDA and evaluated against same-seed Euclidean
frozen baselines.

| gate | roots | teacher | baseline | teacher-only / baseline-only | exact McNemar p | distance delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q=0.25` | 64 | `35/64` | `33/64` | `4 / 2` | `0.6875` | `-0.01675` |
| `q=0.10` | 64 | `35/64` | `33/64` | `3 / 1` | `0.6250` | `+0.01638` |
| `q=0.25` | 128 | `65/128` | `61/128` | `10 / 6` | `0.4545` | `-0.00419` |

Pooling only the two disjoint `q=0.25` panels gives `100/192` versus
`94/192`, teacher-only `14`, baseline-only `8`, exact McNemar `p=0.2863`,
and distance delta `-0.00838` with paired bootstrap 95% CI
`[-0.02513, 0.00577]`.  This is a reproducible positive direction but not a
statistically established teacher gain.  The stricter `q=0.10` arm reverses
the distance direction, showing that excessive abstention removes useful
candidate support rather than solving the transfer problem.

The mechanism diagnosis is now more specific: teacher negative transfer is
not monotonically reduced by making the gate stricter.  A useful gate must
preserve a sufficiently rich supported candidate set while preventing broad
rank replacement.  The `q=0.25` protocol is retained as a candidate repair
for a future preregistered replication, but it is not promoted to a universal
policy or a completed GDD-WM result.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_terminal_gate_q025_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_terminal_gate_q025_fresh64_20261020_cuda_baseline/`
- `le-wm-main/results/gdd_wm_reacher_terminal_gate_q010_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_terminal_gate_q025_fresh128_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_terminal_gate_q025_fresh128_20261020_cuda_baseline/`

## Full/action-only consensus gate — 2026-10-20

To test whether the remaining ranking error could be rejected online, the
runtime now supports `--capacity-bilinear-mode consensus`.  It computes both
the action-only and full predicted-dynamics bilinear rankings and releases a
teacher correction only when their supported argmins agree; disagreement
falls back to the frozen base cost.  The mode is fail-closed and records
`bilinear_consensus_abstentions`.

The corrected implementation passed the full regression suite (`1014 passed,
3 skipped`).  On the matched Reacher 64-root CUDA panel with `q=0.25`, the
two readouts disagreed on all 80 effective bank-ranking calls, so the
consensus controller abstained on all 80.  This is not a teacher gain: it
demonstrates that the current full and action-only geometries have no stable
within-support decision agreement.  The mode is retained as a reusable
diagnostic/fail-closed control, not as the formal controller.

Artifact:

- `le-wm-main/results/gdd_wm_reacher_consensus_q025_fresh64_20261020_cuda_v2/`
- `le-wm-main/experiments/capacity_readout_runtime.py`

## Rank-fusion and base-blend follow-ups — 2026-10-20

Two conservative repairs were tested on the same Reacher 64-root CUDA panel.
Average rank fusion of the full and action-only readouts did not improve the
teacher consequence: it produced `35/64` successes versus `33/64` for the
matched frozen baseline, but final-distance delta was `+0.00383` and exact
McNemar `p=0.6875`.  Thus rank aggregation does not recover the missing task
utility when the underlying geometries disagree.

A fixed `0.5` blend of the mapped teacher cost with the frozen base cost kept
the same success point estimate (`35/64`) and gave a small distance delta of
`-0.00201`, with exact McNemar `p=0.6875`.  This is safer than raw replacement
but still not a significant independent planning gain.

These interventions rule out two simple fixes—averaging incompatible
readouts and uniformly shrinking their magnitude—as sufficient solutions.
The remaining actionable cause is conditional action identifiability: the
teacher needs a state/phase-specific validity signal, not just a global
ranking transform.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_rank_fusion_q025_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_blend05_q025_fresh64_20261020_cuda/`

## Artifact provenance contract repair — 2026-10-20

A provenance audit found that the preceding Reacher q-gate/rank-fusion sweep
had used `gdd_wm_goal_bilinear_teacher_20261020_cuda/capacity_readout.npz`.
That artifact declares `target_mode=root_centered`,
`goal_conditioning=latent_relative`, and
`phase_conditioning=tworoom_rgb_phase_waypoint_action`; it is a TwoRoom chart
and must not be interpreted as Reacher evidence.  Those runs are retained as
diagnostics of the contract failure, but their teacher consequence is
invalidated for scientific Reacher claims.

The Reacher runner now validates before model construction that a capacity
artifact has the absolute-qpos contract:
`target_mode=absolute`, `goal_conditioning=none`,
`phase_conditioning=none`, and `action_frame=raw`.  A mismatched artifact
fails closed with an explicit error instead of producing plausible numerical
scores.  The new contract tests and related runtime tests pass (`38 passed`)
and the full suite remains green (`1014 passed, 3 skipped` before this
documentation-only update).

Using the actually Reacher-compatible artifact
`gdd_wm_repairability_20261018/reacher_history1_vs3/capacity_readout.npz`
on a fresh 64-root CUDA panel gave `35/64` versus the same-seed frozen
baseline `33/64`, with teacher-only `4`, baseline-only `2`, exact McNemar
`p=0.6875`, and final-distance delta `-0.00448`.  This is directionally
positive but not significant; it confirms that contract repair is necessary
and also that it does not by itself solve teacher transfer.

Artifacts:

- `le-wm-main/experiments/run_live_reacher_sequence_candidate_planning.py`
- `le-wm-main/tests/test_live_reacher_sequence_candidate_planning.py`
- `le-wm-main/results/gdd_wm_reacher_correct_contract_history1vs3_q025_fresh64_20261020_cuda/`

## Proposal-relative source-action teacher with frozen q=0.50 gate — 2026-10-20

The next intervention targets the diagnosed failure mode: the teacher can
contain useful action preference information, but unconstrained replacement
of the frozen candidate argmin can trade terminal precision for binary
success.  The controller therefore uses a predeclared proposal-relative
source-action teacher and only permits teacher corrections inside the lower
50% of the frozen base-cost candidate bank.  This is a terminal base-rank
support gate; it is not selected from the held-out planning outcomes in the
reported panel.

Across three root-disjoint CUDA batches (64 roots each), the paired result is
`102/192` successes for the teacher versus `93/192` for the matched frozen
baseline.  The teacher-only/baseline-only discordance is `13/4`, giving exact
McNemar `p=0.04904` and a binary success-rate difference of `+0.046875`.
The paired final-qpos-distance difference is `-0.01189`, with bootstrap 95%
CI `[-0.02465, 0.00100]`; thus the continuous metric is directionally better
but its interval crosses zero.

This result is promising but remains exploratory/sequential because the
`q=0.50` threshold was chosen after the preceding gate sweep.  It is not yet
a confirmatory GDD-WM claim.  The required next test is a fresh root-disjoint
replication with `q=0.50` frozen in advance, followed by an environment-level
transfer test using the same gate semantics.  The result supports a narrower
claim: source-action, proposal-relative geometry plus conservative base-rank
support can repair part of the Reacher teacher's negative transfer.  It does
not establish a D-JEPA teacher, causal latent dynamics, universal transfer,
or an independent frozen-predictor gain across environments.

Artifacts:

- `le-wm-main/experiments/gdd_wm/summarize_reacher_proposal_relative_batches.py`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_aggregate_192_20261020_cuda.json`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_fresh64b_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_fresh64c_20261020_cuda/`

The formal interpretation is therefore: `q=0.50` is a frozen candidate
repair hypothesis awaiting independent replication, while the earlier
contract-mismatch artifacts remain invalidated for Reacher evidence.

## Frozen q=0.50 Reacher replication — 2026-10-20

The threshold was then frozen and evaluated on a fourth root-disjoint batch
(`20269300--20269363`), without changing the artifact, candidate bank, CEM
budget, or RGB-only runtime.  Pooling this batch with the three preceding
independent batches gives `132/256` teacher successes versus `118/256` for
the matched frozen baseline.  Teacher-only/baseline-only discordance is
`19/5`; exact McNemar `p=0.00661`.  The success-rate difference is `+0.05469`.

The paired final-qpos-distance difference is `-0.01874`, with bootstrap 95%
CI `[-0.02987, -0.00731]`.  This is the first held-out Reacher panel in this
workstream where both the binary and continuous planning consequences point
in the same direction under the frozen gate.  The result is still a
single-environment result and therefore does not establish cross-environment
transfer or a universal teacher.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_fresh64d_20261020_cuda/`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_fresh64d_20261020_cuda_baseline/`
- `le-wm-main/results/gdd_wm_reacher_proposal_relative_q050_aggregate_256_20261020_cuda.json`

The next cross-environment experiment must keep `q=0.50` and the
source-action/proposal-relative design frozen.  Only the environment-specific
teacher artifact and its declared action-coordinate contract may change; no
new gate tuning is allowed from the held-out planning outcomes.

## TwoRoom cross-environment boundary test — 2026-10-20

To test whether the Reacher repair was an environment-independent GDD-WM
mechanism, the same frozen `q=0.50` base-rank rule was applied to a
TwoRoom-specific proposal-relative source-action artifact.  The checkpoint,
candidate-bank definition, CEM budget, RGB-only runtime, seed panel
(`20269400--20269463`), and baseline were matched.

The intervention obtained `13/64` successes versus `15/64` for frozen LeWM.
Discordance was teacher-only `3` versus baseline-only `5`, exact McNemar
`p=0.7266`, with paired success difference `-0.03125`.  This is a negative
cross-environment consequence, not a failed run or an artifact mismatch.
Runtime diagnostics show why the Reacher gate is insufficient here: the
TwoRoom teacher changed `720/976` supported bank argmins (`73.8%`) and
abstained on only `122` rows.  A base-cost rank gate therefore constrains
which candidates can be selected but does not certify the action-effect sign
across door/contact phases.

This result changes the method requirement.  A transferable GDD-WM teacher
needs a phase/mode-conditioned action-identifiability gate (or verified local
effect update) in addition to the global q gate.  Reacher's q=0.50 result is
therefore retained as a single-environment positive replication, while the
TwoRoom panel is the explicit adversarial counterexample to universal
transfer.  No cross-environment teacher claim is opened by the Reacher result.

Artifact:

- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_fresh64_20261020_cuda_baseline/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_fresh64_20261020_cuda/paired_summary.json`

## Mode-gated TwoRoom repair attempt — 2026-10-20

To isolate whether door-phase contamination caused the preceding negative
panel, the same RGB door-phase fail-closed rule was enabled for both teacher
and baseline.  The teacher again used the frozen TwoRoom proposal-relative
source-action readout with `q=0.50`; only non-door phases could differ.

The paired panel gave `13/64` teacher successes versus `12/64` baseline
successes.  Discordance was `4/3`, exact McNemar `p=1.0`, with delta
`+0.015625`.  Thus removing the door phase did not produce a reliable
teacher gain.  The failure is broader than a single contaminated phase:
the teacher still changed `772/968` supported bank argmins (`79.8%`) while
the phase gate was active.

The TwoRoom repair path therefore needs a local observed-effect or
mode-conditioned action-identifiability gate, not merely an RGB phase
fallback and not merely a global base-rank quantile.  This is now a concrete
failure boundary for GDD-WM and a required adversarial control for any
cross-environment claim.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_phasegate_fresh64_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_phasegate_fresh64_20261020_cuda_baseline/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_phasegate_fresh64_20261020_cuda/paired_summary.json`

## TwoRoom repair attempts and matched-arm audit — 2026-10-20

Two additional development interventions were run on fresh CUDA roots.
First, adding a fixed `0.5` blend of the frozen base cost to the q=0.50
teacher produced `7/32` successes versus `8/32` for the matched baseline.
Simple shrinkage of the teacher correction therefore does not repair the
negative transfer.

Second, teacher and baseline were both wrapped in the same phase-local RGB
effect-map repair: six-row minimum support, rank-two conditioning, rolling
two-row held-out residual verification, and inverse-effect fallback.  The
teacher hybrid obtained `15/32` and the baseline hybrid `14/32`; discordance
was `1/0`, exact McNemar `p=1.0`.  The point difference is not evidence of an
independent teacher gain.  It shows that the observed local-effect repair can
raise both arms while leaving the teacher contribution unresolved.

During this audit a summarizer bug was found: it compared the intervention's
`hybrid_rgb_fallback` arm to the baseline's `frozen_lewm` arm, which would
have attributed shared RGB repair to the teacher.  The summarizer now
requires an explicit `--baseline-key` and the corrected matched result above
is the only one retained for interpretation.

These results narrow the repair target further.  The next intervention must
rank or gate teacher candidates using an independently verified local effect
chart before execution; post-selection fallback alone cannot demonstrate
teacher-specific action identifiability.

Artifacts:

- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_blend05_dev32_20261020_cuda/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_blend05_dev32_20261020_cuda_baseline/`
- `le-wm-main/results/gdd_wm_tworoom_proposal_relative_q050_verifiedlocal_dev32_20261020_cuda/paired_hybrid_matched_summary.json`
- `le-wm-main/experiments/gdd_wm/summarize_tworoom_repair_consequence.py`

## PointMaze projected GDD fresh CUDA replication — 2026-10-20

After the projection branch and its cross-bank guard were added, the full
regression suite was rerun: `1020 passed, 3 skipped`.  A fresh 20-root CUDA
panel then compared `visual_dynamics` with
`gdd_wm_rgb_projected_guarded` under the same RGB waypoint planner, candidate
bank, frozen JEPA-WM, calibration roots, and held-out roots.

Both arms achieved `20/20` success.  The projected arm had mean final
distance `0.15644` versus `0.16034` for visual dynamics (paired mean delta
`-0.00389`), but its cross-bank gate fallback rate was `0.8333`.  The
projection alpha remained effectively binary (`0` or `1`) rather than
providing a usable continuous teacher correction.  The apparent distance
improvement is therefore not an independently attributable GDD teacher gain;
it is primarily a fail-closed safety wrapper around the visual-dynamics
controller.

This replication fixes the current PointMaze interpretation: projection
guarding prevents negative transfer, but does not validate the teacher or
establish closed-loop benefit.  The diagnosed repair target is now explicit:
the frozen teacher must be trained and calibrated with the same local
waypoint-conditioned action frame used at deployment.  Feeding a global-goal
teacher into a local-waypoint planner remains a covariate shift, and lowering
the guard threshold or adding more projection blending would be invalid
post-hoc tuning rather than a scientific fix.

Artifact:

- `le-wm-main/results/gdd_wm_pointmaze_gdd_projected_guarded_20261020_cuda_fresh20_rerun.json`

The result is a boundary result, not a positive GDD-WM planning claim.

## Waypoint-conditioned teacher direction — existing independent panels

The separate TwoRoom waypoint-target panels provide the correct next design
direction.  With the same frozen waypoint-target teacher artifact across
three independent 32-root panels, the intervention obtained `26/96`
successes versus `19/96` for the matched baseline; teacher-only versus
baseline-only discordance was `12/5`, exact McNemar `p=0.14346`.  This is a
positive point estimate but not a confirmatory result.  It is nevertheless
more informative than the PointMaze projection rerun because the teacher's
task target is aligned with the deployment phase/waypoint rather than with a
single global goal.

The next valid escalation is therefore not another threshold sweep.  It is a
fresh root-disjoint waypoint-conditioned panel with the same artifact and
predeclared action bank, followed by an independent closed-loop consequence.
Only if that replication survives calibration authorization should it be
connected to GDD distillation.  The current combined result remains
exploratory and does not establish cross-environment transfer.

Artifact:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/tworoom_waypoint_teacher_3panel_summary.json`

## PointMaze waypoint-conditioned GDD authorization — 2026-10-20

The waypoint-conditioned projection was then tested under the complete
train/calibration/test separation.  This closes the previously observed
global-goal versus local-waypoint mismatch without changing the frozen JEPA
predictor or adapter.  The h1 package uses the same five-step pulse horizon
as the online planner, and the authorization gate was fitted on five roots
disjoint from the twenty held-out test roots.

The calibration-only authorization rejected the GDD teacher.  On the
held-out panel, visual dynamics reached `20/20` with mean final distance
`0.13908`; the requested guarded waypoint teacher also reached `20/20`, but
with mean final distance `0.15290` and gate fallback rate `0.3809`.  Because
authorization was false, this arm is not evidence of a teacher gain: the
runtime correctly remains fail-closed and reports the requested method plus
the authorization fallback marker.

The corresponding unguarded arm is a useful adversarial control: it reached
only `2/20` on the same clean held-out protocol, with mean final distance
`1.13530`.  Thus the guard prevents severe negative transfer, but it does not
recover independent planning value.  The same qualitative result was seen in
the strict h4/terminal and h1 waypoint panels: conditioning the target and
matching the terminal horizon removes the obvious covariate-shift and label
horizon bugs, yet the frozen learned action-effect ranking remains unreliable.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_waypoint_projection_h1_20261020_cuda/`
- `le-wm-main/results/gdd_wm_pointmaze_waypoint_teacher_h1_20261020_cuda_cleanheldout20_threearm.json`
- `le-wm-main/results/gdd_wm_pointmaze_waypoint_teacher_h1_20261020_cuda_cleanheldout20_calibratedgate_v2.json`
- `le-wm-main/results/gdd_wm_pointmaze_waypoint_teacher_h1_20261020_cuda_authorized_cleanheldout20.json`

### Updated diagnosis

The combined intervention isolates the remaining failure more sharply than
the earlier global-goal experiment:

1. waypoint conditioning fixes the deployment target contract;
2. horizon matching fixes the terminal-label versus pulse mismatch;
3. calibration-only authorization prevents the teacher from being released
   when it regresses;
4. unguarded failure persists, so projection geometry is not sufficient to
   identify within-support action effects.

The next repair must preserve the deployment action support and verify the
local action-effect sign before ranking candidates.  A new positive result
would require a frozen support-preserving teacher, an independently calibrated
mode/effect gate, and a matched held-out planning consequence.  Lowering the
fallback threshold, counting the shared RGB controller, or selecting on the
held-out distance would not repair this scientific failure.

Current status: **GDD-WM teacher closed-loop authorization: rejected**;
**fail-closed safety behavior: validated**; **independent GDD planning gain:
not yet demonstrated**.

## Support-preserving release repair — 2026-10-20

The next intervention removed the remaining cross-bank confound.  The new
runtime method `gdd_wm_rgb_support_preserving_guarded` lets the frozen GDD
student rank its paired candidates, but maps the selected first pulse to the
nearest action in the same fixed visual-dynamics bank before release.  The
mapping is deterministic and non-learned; its distance is recorded in every
trace.  A strict non-degradation gate then releases the mapped action only if
its calibration visual-dynamics score is no worse than the best visual bank
action; otherwise it falls back.

On the clean 20-root PointMaze panel, the initial support-preserving version
still regressed: final distance `0.17386` versus `0.13908` for visual
dynamics, despite `20/20` success.  This confirms that merely projecting onto
the deployment action support does not repair the learned preference.

The strict non-degradation version removed that regression, but only by
falling back on `418/455` decisions (`91.87%`).  Its held-out result was
exactly the visual baseline: `20/20`, final distance `0.13908`.  The
calibration-only outcome rule alone would have called this teacher authorized,
because the five calibration roots were tied after fallback.  That was a
false authorization: the method had almost no released teacher actions.

The authorization contract is therefore strengthened in code with a fixed
minimum teacher release rate of `0.20` for this method.  A fallback-dominated
arm is now rejected even when its shared fallback controller matches the
baseline.  This separates three outcomes that were previously conflated:

- negative transfer: support-preserving release without strict gating;
- safe fallback: strict gating with no independent teacher action;
- teacher gain: not observed in this panel.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_support_preserving_20261020_cuda_cleanheldout20.json`
- `le-wm-main/results/gdd_wm_pointmaze_support_preserving_strict_20261020_cuda_cleanheldout20.json`
- `le-wm-main/results/gdd_wm_pointmaze_support_preserving_releasegate_fixed_20261020_cuda_cleanheldout20.json`

In the final rerun the release-rate gate correctly rejected the method before
the held-out consequence (`authorized=false`): calibration release rate was
`0.1689`, below the fixed `0.20` minimum.  The held-out report still shows the
requested arm executing as the declared visual fallback with `20/20` and final
distance `0.13908`.  The rejection is now machine-readable rather than
inferred from the fallback count after the fact.

The next required experiment is a newly trained adapter whose candidate
generation, action chart, waypoint conditioning, and one-pulse deployment
bank are identical at train and test time.  Reusing the current paired-order
adapter with a release wrapper has now been exhausted as a repair path; any
future positive result must come from support-matched training plus an
independent held-out consequence, not from further fallback tuning.

## Exact support-matched projection consequence — 2026-10-20

The support-matched package was then built and evaluated.  It expands each
source context into the exact 32-action visual fixed bank used by the live
planner, encodes each candidate as the same repeated 20-step pulse, and fits
the projection only on train-root execution costs.  The package contains
`2560` records over `80` root/context groups and has no runtime teacher-rank
or candidate-ID input.

The clean held-out consequence still failed to release a teacher: visual
dynamics reached `20/20` with final distance `0.13908`; the support-matched
guarded arm also reached `20/20` and exactly the same final distance, but
fell back on `98.24%` of gate decisions.  Its calibration release rate was
only `0.0235`, so the corrected release-rate authorization rejected it before
test deployment.  The requested arm therefore executed as the declared visual
fallback, not as a positive support-matched teacher.

Artifacts:

- `le-wm-main/experiments/gdd_wm/build_pointmaze_support_matched_projection_package.py`
- `le-wm-main/results/gdd_wm_pointmaze_support_matched_projection_20261020_cuda/`
- `le-wm-main/results/gdd_wm_pointmaze_support_matched_consequence_releasegate_20261020_cuda_cleanheldout20.json`

This is a stronger diagnosis than the earlier nearest-action repair.  The
negative result persists after matching the candidate family, action chart,
active waypoint, pulse horizon, and deployment support.  The remaining
bottleneck is therefore the learned latent/action-effect representation (and
its train-to-root generalization), not merely cross-bank compatibility.  The
next valid intervention must change the representation or train the adapter
with observed successor effects in this exact support; another runtime gate
or projection wrapper would only reproduce the same fallback boundary.

## RGB successor-supervised adapter audit — 2026-10-20

To test whether the missing ingredient was simply successor supervision, a
separate adapter was trained from the two root-disjoint RGB successor joins.
Each row used the frozen source latent and complete action sequence to predict
the realized successor-latent delta; the latent-to-terminal-XY readout was fit
only on train-root realized terminal states.  No calibration or outcome row
was used for fitting.

The basis-action adapter produced calibration mean ranking regret `0.1543`
with exact top-1 agreement `12.5%`, and outcome regret `0.1685` with exact
top-1 agreement `12.5%`.  Replacing the fixed action basis by the raw 40-D
sequence did not repair it: outcome regret was `0.1719`, also with `12.5%`
top-1 agreement.  Thus the observed successor supervision itself does not
recover reliable within-bank action preference for this frozen latent.

Artifacts:

- `le-wm-main/experiments/gdd_wm/train_rgb_successor_latent_adapter.py`
- `le-wm-main/results/gdd_wm_rgb_successor_latent_adapter_20261020_cuda/adapter_fixed_readout.pt`
- `le-wm-main/results/gdd_wm_rgb_successor_latent_adapter_20261020_cuda/adapter_fixed_readout.json`
- `le-wm-main/results/gdd_wm_rgb_successor_latent_adapter_20261020_cuda/adapter_raw_fixed_readout.json`

This closes the “more successor labels” repair for the current one-frame
latent and paired candidate panel.  The remaining repair must add information
that is absent from the representation—causal history, contact/mode state, or
an active observed-effect update—rather than merely increasing adapter
capacity or changing the action feature basis.

## Teacher artifact repair and independent TwoRoom consequence — 2026-10-20

The next audit separated an artifact/interface failure from a scientific
failure.  The earlier TwoRoom capacity artifact declared latent-relative goal
conditioning but did not export the required `bilinear_goal_mean` and
`bilinear_goal_components`.  The runtime correctly rejected its full
predicted-dynamics branch; this was not evidence that the branch had been
tested.  A fresh CUDA audit regenerated the complete goal-relative chart from
the two root-disjoint RGB panels, with train-only fitting and calibration-only
selection:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_vision_door_teacher_fullchart_64shared/capacity_audit.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_vision_door_teacher_fullchart_64shared/capacity_readout.npz`

The corrected artifact is complete, but its predeclared gate still selects
`action_only`; the predicted-dynamics term is not authorized.  The paired
closed-loop run on seeds `20262900..20262931` therefore gives the following
diagnostic result:

| arm | success | runtime interpretation |
| --- | ---: | --- |
| frozen LeWM with gate-authorized action-only artifact | `13/32` | teacher readout active, but below the frozen reference |
| frozen LeWM + online RGB effect verification | `18/32` | observed-effect fallback/update active |

For the second run, the frozen reference inside the same artifact is `15/32`;
the verified arm has 4 paired wins, 1 loss and 27 ties (`+3/32`), with
`182/1217 = 14.95%` fallback decisions.  The paired artifact is:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_verified_rgb_hybrid_20262900_32_clean/planning.json`

This is an independent TwoRoom closed-loop consequence, not evidence for an
independent GDD-WM teacher gain.  It identifies the actionable repair boundary:
the frozen latent teacher does not yet carry a stable decision preference,
whereas short-pulse RGB re-observation and local effect verification can
improve the same frozen planner.  Future teacher work must therefore train or
distill from verified observed effects and must beat the frozen reference on a
new paired panel; simply enabling an exported latent bilinear term is not a
valid repair.

## Full goal-relative dynamics stress test — 2026-10-20

The complete goal-relative chart was also run with an explicit
`full_predicted_dynamics` override.  This is an exploratory stress test only:
the train/calibration authorization gate had rejected this branch, so it is
not a released controller.  Unlike the earlier malformed-artifact attempt,
the runtime actually evaluated the full branch on `496` candidate rows and
changed the bank argmin on `81.85%` of evaluated rows.

On the same 32-seed TwoRoom panel it achieved only `5/32` successes, compared
with `13/32` for the newly regenerated gate-authorized action-only artifact and
`15/32` for the corresponding frozen reference in the verified RGB run.  This
is a genuine negative intervention, not an inactive-interface artifact.  It
locates the remaining failure more sharply: predicted-latent-delta interaction
terms are not merely unhelpful; when allowed to control ranking, they can
overwrite a useful frozen cost with unstable preferences.

Artifact:

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_vision_door_teacher_full_dynamics_exploratory_20262900_32/planning.json`

The formal repair rule is now fixed: no latent dynamics interaction may enter
closed-loop planning unless it passes train cross-fit, calibration, and an
independent paired consequence.  The positive repair target remains an
observed-effect preference distillation or online local update, not a stronger
unverified latent bilinear term.

## Probe-budget intervention — 2026-10-20

The root-local audit suggested that the four-probe online protocol was
under-sampling the action chart: with outcome-free coverage probes, increasing
the diagnostic budget from 3 to 8 reduced offline local-effect regret from
`14.69` to `1.70` on 64 TwoRoom roots, while the frozen teacher regret was
`32.85` in the same 8-probe audit.  This motivated a real code intervention,
not a post-hoc threshold change: `active_probe_actions` now accepts a fixed
even direction count, the default remains the historical 4, and the protocol
records `active_probe_directions`.

The independent CUDA PointMaze run with 8 directions completed on 20 held-out
roots.  It reached `20/20`, mean final distance `0.14708`, versus `0.14850`
for visual dynamics, but used `8` active probe blocks per episode and fell
back on `90.30%` of planning gate decisions.  This is a small distance
improvement with no success-rate improvement and is not an authorized teacher
gain; it demonstrates that offline probe-budget gains do not automatically
carry into online planning under the current release gate.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_root_local_effect_ranker_probe3.json`
- `le-wm-main/results/gdd_wm_pointmaze_root_local_effect_ranker_probe8.json`
- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_jacobian_20261020_cuda_fresh20_directions8`
- `le-wm-main/experiments/gdd_wm/run_active_probe_local_jacobian_cuda.sh`

The next repair should therefore optimize the information-action protocol and
the release criterion jointly: an active probe is useful only if its observed
effect changes the subsequent action choice and the resulting planner retains
an independent benefit after probe cost.  More probes alone are not a valid
solution.

## Local-direct online ranking intervention — 2026-10-20

The previous PointMaze local-Jacobian arm was not a clean test of whether
observed RGB effects could drive decisions: it compared the local chart to the
visual-dynamics score at every step and fell back whenever the difference
exceeded `0.10`.  On the four-probe panel this made `92.90%` of decisions
fallback.  To separate an over-conservative gate from an invalid local chart,
the runtime now exposes a predeclared
`effect_adapter_rgb_online_local_direct` arm.  It ranks the fixed action bank
with the episode-local RGB observed-effect affine chart only after rank-two
support is trusted; every released action is then checked against its next RGB
successor, and a failed check quarantines the chart.  The arm does not load or
query the frozen effect-policy checkpoint.

The clean 20-root CUDA consequence was negative even under this direct test:

| arm | probes | success | mean final distance | verification | quarantine |
| --- | ---: | ---: | ---: | ---: | ---: |
| visual dynamics | 0 | `20/20` | `0.14850` | — | — |
| local direct | 4 | `16/20` | `0.52394` | `95.09%` | `80%` |
| local direct | 8 | `17/20` | `0.32377` | `94.74%` | `70%` |

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_direct_20261020_cuda_fresh20_directions4/planning_report.json`
- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_direct_20261020_cuda_fresh20_directions8/planning_report.json`
- `le-wm-main/experiments/run_live_pointmaze_planning.py`

The intervention rules out two explanations for the earlier negative result:
it is not only caused by the visual-consistency threshold, and it is not
removed by doubling probe coverage.  Successor verification is high, but the
chart is quarantined in most episodes and direct ranking damages route-level
progress.  The current failure boundary is therefore phase/support transfer:
an observed local effect chart is reliable for the immediately verified pulse
but is not a sufficient route-conditioned action preference after waypoint or
obstacle interactions.  This arm is not an authorized GDD-WM teacher and is
not a frozen-predictor gain.  The next valid repair must add phase-aware chart
reset or obstacle/contact-conditioned release and beat the same visual arm on
a new paired root panel.

## Phase-reset and phase-local ranker audits — 2026-10-20

The PointMaze local-direct failure was next tested as a phase-transfer
hypothesis.  The new `effect_adapter_rgb_online_local_phase` arm discards its
RGB local effect chart whenever the active RGB-maze waypoint changes, executes
the same fixed four-direction probe budget in the new phase, and only then
returns to direct local ranking.  This is not a threshold relaxation: both
the chart reset and the re-probe actions are recorded in the held-out trace.

On the same 20-root CUDA panel, phase reset was strongly negative: visual
dynamics remained `20/20` with mean final distance `0.14850`, while phase
reset reached `4/20` with mean final distance `1.40976`.  It consumed an
average of `7.0` extra phase probes per episode (in addition to the initial
four), and `35` route-phase transitions were observed.  The intervention
therefore rejects the explanation that reuse of an old chart is the only
cause of negative transfer: repeated probing itself changes the route state,
and a phase-local affine chart is still insufficient for obstacle-aware
multi-step choice.

- `le-wm-main/results/gdd_wm_pointmaze_active_probe_local_phase_20261020_cuda_fresh20_directions4/planning_report.json`

TwoRoom provides genuine headroom, so the existing phase-local observed-RGB
effect ranker was separately audited on a fresh 32-seed CUDA panel.  With the
ranker enabled, the verified-RGB hybrid reached `16/32` versus frozen LeWM
`14/32`; the ranker reported `126` supported candidate rows.  A strict
same-seed control with the ranker disabled also reached exactly `16/32`, with
the same three hybrid-only wins and one frozen-only win.  Thus the apparent
`+2/32` improvement is the already-known verified-RGB hybrid effect, not an
independent ranker contribution.

The runtime now records local-ranker decision and argmin-change counts.  The
diagnostic rerun showed that the ranker changed `18/126` supported candidate
argmins (`14.29%`), yet did not change the paired episode outcome.  This is a
stronger and more useful boundary than an inert-scorer explanation:
candidate-level preference changes currently do not carry through the CEM
selection and receding-horizon trajectory into terminal success.  Future
teacher authorization must therefore report both candidate argmin change and
an executed-action/closed-loop consequence; a nonzero local ranking change is
not sufficient.

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_phase_local_ranker_20263000_32/planning.json/planning.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_verified_rgb_control_20263000_32/planning.json/planning.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_phase_local_ranker_diagnostic_20263000_32/planning.json/planning.json`

Current conclusion: phase-aware local maps and successor verification are
useful safety diagnostics, but neither has yet converted into a standalone
frozen GDD teacher.  A valid next teacher intervention must cause a measured
candidate argmin change and then improve an independently paired closed-loop
panel; score activity or fallback-matched success is insufficient.

The complementary full-bank intervention was also completed.  The original
`candidate_bank_iterations=7` injected supported candidates into only the
first seven of eight CEM iterations.  Keeping the same fixed bank through all
iterations raised local candidate argmin changes to `26/168` (`15.48%`), but
the same fresh 32-seed panel reached only `12/32` for both the ranker hybrid
and its no-ranker verified-RGB control.  Thus the missing carry is not merely
caused by the final Gaussian iteration overwriting bank preferences.

- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_phase_local_ranker_fullbank_20263100_32/planning.json/planning.json`
- `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261020/tworoom_verified_rgb_fullbank_control_20263100_32/planning.json/planning.json`

This closes the current CEM-support repair family: local RGB effect ranking
can alter supported candidate preferences, but neither the seven-iteration
nor all-iteration bank protocol yields an independent closed-loop benefit.
The next GDD-WM repair must change the learned task representation or train a
policy-level action selector from verified effects; further solver-bank
wrapping would not address the observed mechanism.

## Fresh GPU teacher closed-loop confirmation — 2026-10-20

The next independent test isolates the teacher-to-planner path that remains
deployable after the TwoRoom local-ranker boundary.  It uses the already
frozen, train-root-only proposal-relative capacity readout on Reacher, with
the frozen LeWM checkpoint, `capacity17` candidate bank, `best` final action,
and identical CEM budget.  No new root outcome entered artifact selection.

The initial 128-environment vectorized launch exceeded the stable CUDA
parallel capacity and wrote no result artifact.  The protocol was therefore
executed as four disjoint 32-root CUDA batches, preserving the exact controller
and root pairing.  On the new 128 roots (`20265100`–`20265227`):

| controller | successes | mean terminal qpos distance | mean minimum qpos distance |
| --- | ---: | ---: | ---: |
| frozen LeWM | `67/128` | `0.20277` | `0.09643` |
| frozen train-only decision readout | `70/128` | `0.18491` | `0.09203` |

The stratified paired effects (readout minus frozen LeWM) are:

- terminal distance: `-0.01786`, 95% bootstrap CI `[-0.03371, -0.00223]`;
- minimum distance: `-0.00440`, 95% bootstrap CI `[-0.00886, -0.00032]`;
- success: `+0.02344` (`7` readout-only versus `4` baseline-only wins), 95%
  bootstrap CI `[-0.02344, 0.07031]`.

The intervention was active: it changed the supported bank argmin on `71/72`
planner rows, without low-support abstentions.  Thus the continuous closed-loop
consequence is positive on a fresh GPU-only panel, while the binary-success
effect remains statistically unresolved and must not be promoted to a success
claim.  This is the first current GDD-WM workstream result that clears an
independent closed-loop *continuous-control* gate; it is still a frozen
train-only readout control, not D-JEPA-teacher validation, a predicted-dynamics
gain, or cross-environment/universal evidence.

Artifacts:

- `le-wm-main/results/gdd_wm_reacher_teacher_confirmation128_20261020_cuda/aggregate.json`
- `le-wm-main/results/gdd_wm_reacher_teacher_confirmation128_20261020_cuda/comparison_batch{0,32,64,96}.json`
- `le-wm-main/results/gdd_wm_reacher_teacher_confirmation128_20261020_cuda_{baseline,teacher}_batch{0,32,64,96}/planning.json`

## PointMaze one-block effect-contract repair — 2026-10-20

The prior online effect teacher predicted a 20-step terminal effect while the
live controller executed one 5-step block and immediately re-observed.  This
is a causal deployment-contract mismatch, not a calibration-threshold issue.
The runtime now rejects a teacher unless its declared action horizon equals
the execution horizon and its candidate family is explicitly
`runtime_one_block_pulse_v1`.  A new 100-root, 32-action, 5-step GPU-generated
counterfactual bank was fit on 54 train roots and calibrated on 14 disjoint
roots; the 32 outcome roots remained outside fitting and calibration.

The latent-source version did not solve the problem: its unguarded controller
reached `3/20` success, while the gate released `0/697` selections because the
deployment latent-support distance (about `13`) was far beyond the train-bank
threshold (`1.26`).  This identifies a second causal defect: the offline
source-latent export and live RGB encoder use incompatible source charts.

An explicitly labelled diagnostic control replaced that source chart with
RGB-derived position/velocity at deployment (the simulator kinematics are
only its offline training proxy).  It is **not** latent-only GDD-WM evidence.
It establishes that the remaining issue is chart alignment rather than the
one-block effect target: calibration direction cosine improved from `-0.243`
to `+0.864`, and effect-error p90 from `0.304` to `0.047`.  On the unopened
20-root GPU panel, it achieved `20/20` success with terminal distance `0.1625`,
versus visual dynamics `20/20` and `0.1485`.  Thus it restores stable local
action-effect control but does not beat the strong visual-dynamics baseline.

HTSC correctly remained fail-closed: only eight support-qualified calibration
roots were available, so the valid finite-sample conformal rank was
unavailable and it released zero actions; its outcome exactly matched the
visual-dynamics fallback.  Neither fallback parity nor the proxy controller
is evidence of a GDD-WM teacher gain.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/bank/manifest.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_summary.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/planning_report.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_rgb_kinematics_proxy_summary.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/planning_report_rgb_kinematics_proxy.json`

### Live RGB source-chart completion

The preceding proxy control identified the remaining deployability defect:
the offline source-latent export did not use the same renderer, two-frame
context, transform, and encoder invocation as the live controller.  The new
`rebuild_pointmaze_effect_rgb_source_chart.py` reconstructs only train and
calibration sources through that exact live chart; outcome rows remain
untouched.  It also exposed and corrected a second contract bug: a history-1
teacher needs one no-op block to reach exported frame 5 while still supplying
two frames to the encoder, not two no-op blocks (frame 10).

With this repair, the frozen-latent one-block teacher has 13 support-qualified
calibration roots and a valid alpha=0.1 finite-sample root bound of `0.0`.
On the same unopened 20-root CUDA panel, the calibrated HTSC arm released
`74.43%` of decisions and obtained terminal distance `0.15794`, compared with
`0.16006` for visual dynamics; both reached `20/20` success.  The unguarded
teacher was worse (`0.16401`), so the tiny continuous advantage is a
selective-planning result, not a general teacher superiority claim.  This
validates a non-vacuous source-chart + horizon contract repair, but requires a
fresh larger panel and an independent environment before it can support a
cross-environment GDD-WM claim.

Additional artifacts:

- `le-wm-main/experiments/gdd_wm/rebuild_pointmaze_effect_rgb_source_chart.py`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/live_rgb_source_chart/report.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent_summary.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/planning_report_live_rgb_latent_frame5.json`

### External renderer-transfer confirmation

The repository has no stored RGB files for PointMaze roots 1900--1999, but it
does retain their simulator reset states.  To avoid silently treating those
states as a visual test set, the 1900--1931 confirmation panel renders them
only through the renderer fitted from the 1800--1815 calibration RGB roots.
This is external **renderer-transfer** evidence, not stored-RGB validation.

Across 32 unseen roots, the corrected latent HTSC controller released `74.64%`
of decisions, achieved terminal distance `0.14238`, and retained `32/32`
successes.  The shared-budget visual-dynamics baseline achieved `0.15658` and
`32/32` successes, a `-0.01420` continuous terminal-distance difference in
favor of selective latent effect control.  The unguarded teacher was nearly
identical (`0.14245`); this means the present external result validates the
repaired teacher path and non-vacuous gate, but does not isolate an additional
risk--utility benefit of HTSC over its unguarded version.  It also cannot
support a claim about real held-out RGB observations.

Artifact:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/external1900_1931_renderer_transfer.json`

### Stored-RGB remainder check

The untouched stored-RGB roots 1888--1899 were run after freezing the same
teacher, horizon, chart, gate and budget.  All methods again reached `12/12`
success.  The unguarded teacher reached `0.14460` terminal distance, while
HTSC reached `0.15786` and visual dynamics `0.15648`.  Combining this fixed
remainder with roots 1868--1887 gives a 32-root stored-RGB HTSC versus visual
dynamics terminal-distance difference of approximately `-0.00081`; binary
success is saturated throughout.  This removes the apparent 20-root positive
as a reliable stored-RGB gain claim.  The supported statement is narrower:
the contract-repaired latent teacher is non-vacuously deployable; the current
HTSC rule has not yet established a stable stored-RGB risk--utility advantage.

Artifact:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/stored_rgb_1888_1899_confirmation.json`

### Calibration-frozen action-consistency gate

The prior HTSC gate certified only latent/action support and a fixed-bank
regret bound.  It could therefore release an effect-teacher action without
testing whether that particular action was preferable to the independent,
RGB-only visual-dynamics candidate at the same live state.  The live runner
now records that action-level disagreement and visual regret, and supports a
strictly disjoint calibration selector for the consistency tolerance.  Its
candidate grid includes `-1.0`, an explicit visual-only control that forces
every effect action to abstain.  A releasing threshold may win only by
improving calibration success or terminal distance; lower fallback rate alone
cannot authorize it.

On the 14 effect-policy calibration roots (1854--1867), every releasing
threshold had the same saturated success (`14/14`) and did not improve final
distance over visual-only control.  The frozen selector therefore chose `-1.0`.
On all 32 untouched stored-RGB outcome roots (1868--1899), the calibrated
consistency arm exactly matched visual dynamics: `32/32` success and mean
terminal distance `0.15872`.  This is a useful stability repair—it prevents a
spurious release claim—but it is not a teacher gain.  In the same panel, the
unguarded teacher achieved `0.15673`, while legacy HTSC reached `0.15791`.
The resulting diagnosis is sharper: globally calibrated support/bound and
visual-consistency gates cannot yet identify the locally useful teacher
actions.  The next method must estimate conditional action-transfer risk from
disjoint calibration data, rather than relax a global threshold.

Artifact:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/stored_rgb_1868_1899_consistency_calibrated.json`

### Matched observed-local Jacobian audit

The observed-local PointMaze controller had two hidden deployment defects.
First, its active-probe trace could create a frozen residual even though the
observed-only Jacobian has no frozen effect policy; this could stop the intended
probe sequence before the chart was identified.  Second, its release gate
subtracted a local predicted terminal *distance* from a visual controller score
that also contained velocity and action regularizers.  That quantity has no
common task unit, so it could release a locally bad action.

The repaired implementation separates frozen-residual and observed-only
controllers, records all observed local probes, and compares both proposals in
successor-distance task units.  It also adds a `visual_dynamics_active_probe`
control which executes exactly the same four predeclared RGB probes before
using visual dynamics.  This separates information-action cost from local-chart
utility.

On roots 1854--1867, calibration-only selection chose local visual tolerance
`0.20` (`14/14` success; final distance `0.14686`), improving over the matched
all-fallback control (`0.15739`).  On the 32 untouched stored-RGB roots
1868--1899, the corrected local Jacobian reached `32/32` success and terminal
distance `0.14583`, an improvement over the pre-repair `0.15506`, but it still
lost to the same-budget active-probe visual control (`32/32`, `0.13544`).  The
paired terminal-distance difference was `+0.01039` (local Jacobian minus
visual-probe), with 10 wins, 15 losses, and 7 ties.  Thus the implementation
and task-utility contract are repaired, but cross-root local-chart transfer is
not yet sufficiently stable to authorize a planning-gain claim.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/stored_rgb_1868_1899_verified_local_probe_repaired_v2.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/stored_rgb_1868_1899_local_jacobian_utility_calibrated.json`

### Cross-fitted local-chart reliability check

The local Jacobian's rank gate only establishes that an affine chart can be
fit; it does not establish action-effect transfer.  The controller now reports
the leave-one-out observed-pulse error of its local chart and can fail closed
when that cross-fitted error exceeds a declared tolerance.  This is a
conditional transfer-risk diagnostic, distinct from both latent support and
the visual score agreement gate.

On the disjoint 1854--1867 calibration roots, finite local-chart LOO errors
had median `0.1498` and ranged from `0.0287` to `0.2438`.  A `0.20` gate
reduced release from about `1.6%` to `1.2%`, but worsened calibrated terminal
distance from `0.15138` to `0.15248`; both were still better than the
same-budget visual-probe control (`0.15739`).  The result validates that the
risk is measurable, but does not authorize the LOO threshold as the formal
controller because it provides no incremental calibration benefit.  The
remaining repair target is a more informative probe design / task-conditioned
local chart, not a stricter gate alone.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/local_jacobian_loo_calibration_probe.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/local_jacobian_loo020_calibration_probe.json`

### Goal-aligned information-probe audit

The cardinal probe protocol contains a reverse-goal pulse, so its observed
local chart could have failed simply because information collection consumed
task progress.  A new task-aligned four-pulse design keeps every action in the
positive goal half-plane while retaining rank-three affine excitation.  The
comparison includes the exact same task-aligned probes followed by visual
dynamics, so information-action cost is controlled.

The first dispatch smoke failed closed because the new method was omitted from
the local-chart initializer; it is retained only as an invalid smoke artifact.
After the initializer repair, the 14 disjoint calibration roots show that
goal-aligned probing itself helps the visual controller (`0.13372` final
distance versus `0.15739` for cardinal probing), but the observed local chart
still degrades it (`0.14321`, all methods `14/14` success).  Thus reverse-goal
probe cost was a real confound but not the remaining cause of local-policy
failure.  The current affine chart is not a stable task-ranking model across
roots; no final outcome panel or positive claim is authorized for this arm.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_probe_calibration.json` (invalid dispatch smoke)
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_probe_calibration_v2.json`

### Velocity-residual local-effect repair

The remaining local-chart failure was traced to state confounding: serial
probes change velocity, while the original affine fit models only
`effect(action)`.  The repaired arm fits observed effect *residuals* after
subtracting the RGB-calibrated, velocity-conditioned visual-dynamics effect,
then adds that baseline back when ranking the current candidate bank.  It uses
the goal-aligned probe design and the same fixed `0.10` visual-consistency
tolerance as the preceding calibration protocol.

On calibration roots 1854--1867, the residual chart reached terminal distance
`0.13211`, compared with `0.13372` for the exact same task-aligned probes plus
visual dynamics and `0.14321` for the non-residual local chart.  On the 32
untouched stored-RGB roots 1868--1899, the residual controller retained
`32/32` success and reached `0.14809` versus `0.15144` for its same-budget
visual control (paired mean delta `-0.00335`; 14 wins, 4 losses, 14 ties).
The paired bootstrap 95% interval `[-0.01943, 0.01459]` crosses zero, so this
is a replicated directional improvement and a causal mechanism repair, not a
statistically certified planning-gain claim.  The controller also remains an
RGB observed-local recovery method; it does not establish frozen LeWM teacher
utility or universal transfer.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_calibration.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_outcome32.json`

### Renderer-transfer boundary

The same frozen residual protocol was evaluated on roots 1900--1931 using a
dataset-matched renderer fit only from the 1800--1815 calibration RGB roots;
these roots have reset states but no stored RGB and are therefore explicitly
renderer-transfer—not real held-out RGB—evidence.  The residual chart reached
`0.16129` terminal distance versus `0.15780` for the matched visual controller
(both `32/32` success).  It consequently does not transfer across this visual
source change.  The valid positive scope remains same-source stored-RGB local
recovery; renderer invariance, cross-environment utility, and universal frozen
world-model claims remain closed.

Artifact:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_renderer_transfer32.json`

### Local-chart uncertainty gate: calibration rejection — 2026-10-20

The renderer-transfer boundary suggested that the remaining local chart error
could be action-space extrapolation rather than another missing support
threshold. We therefore added an explicit observed-chart uncertainty term: the
local affine chart computes its leave-one-out (LOO) effect residual and
multiplies it by ridge-design leverage for each candidate action. The local
ranker may minimize

\[
  \widehat c(a) + \lambda\,\widehat\sigma_{\mathrm{LOO}}(a),
\]

where \(\lambda\) is selected exclusively on disjoint calibration roots. An
unidentified chart has infinite uncertainty and cannot be released by this
term. This is a robust local-identification intervention inspired by online
system identification / adaptive MPC; it is not a learned-world-model
certificate.

The CUDA calibration comparison used roots `1854--1867`, with the same RGB
task-aligned probe budget and visual-dynamics probe control. The point chart
(`lambda=0`) reached final distance `0.13211`; the robust alternative
(`lambda=1`) reached `0.13291`, while the matched visual control reached
`0.13372`. The larger penalty reduced local releases from `3.50%` to `0.39%`
but worsened the calibration task metric. The pre-outcome selection therefore
froze `lambda=0`; this rejects the hypothesis that a missing conservative
release threshold was the cause of the instability.

On untouched stored-RGB roots `1868--1899`, the frozen result reproduces the
same directional residual benefit (`0.14809` versus `0.15144`, both `32/32`
success) with a local release rate of `3.90%`. This does not strengthen the
previous inference: the paired uncertainty interval still crosses zero, and
the external renderer-transfer failure remains. Its value is causal
localization: with the current probe budget, uncertainty gating can safely
abstain but cannot manufacture missing action-effect information.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_uncertainty_calibration_scale0.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_uncertainty_calibration_scale1.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_uncertainty_outcome32_frozen_scale0.json`

### Conditional local-transfer audit and release-rate boundary — 2026-10-20

The residual controller's small terminal difference could have meant either
that its local actions were not truly better, or that real one-block gains were
too sparse to dominate a saturated terminal metric. The live runner now has an
explicit evaluation-only audit: after the RGB-only controller commits to its
proposal, a simulator clone scores the local and visual proposals for one
block. No counterfactual state, label, or cost is exposed to the controller;
the audit is recorded solely for calibration and held-out mechanism analysis.

On calibration roots `1854--1867`, there were 257 chart-identifiable decisions.
The `0.10` visual-task tolerance released 9 local proposals; all 9 had lower
true one-block task cost than the visual proposal (mean true local-minus-visual
regret `-0.01938`). On independently run stored-RGB roots `1868--1899`, the
same frozen policy released 20 of 513 decisions; 18/20 had lower true one-block
cost, with mean regret `-0.01306`. Predicted local-versus-visual regret had
Pearson correlation `0.48` on calibration and `0.50` on outcome, whereas chart
LOO error was effectively uninformative. This is the first direct evidence
that the residual chart can selectively repair *local action ranking* under
the current RGB interface. It remains an evaluation-only causal audit, not a
runtime privileged planner.

The audit also rejects two tempting but invalid repairs:

1. A strict predicted-improvement-only gate (`tolerance=0`) worsened
   calibration terminal distance to `0.13856`; prediction sign alone does not
   preserve the useful local proposals.
2. Increasing the release tolerance to `0.20` increased releases but worsened
   calibration terminal distance to `0.13884` (versus `0.13211` at `0.10`).
   More locally favorable one-block actions do not automatically improve the
   multi-step route objective.

Velocity-conditioned residual features and route-phase chart resets were also
implemented. Both fail their calibration comparison: the former is
data-starved under four task-aligned pulses (`0.13939`), while phase reset
spends an average 5.71 extra probe blocks and reaches `0.15408`. During that
intervention, a real phase-change implementation defect was repaired: the
waypoint comparison passed an invalid argument sequence to `np.allclose`, so
the prior phase-reset branch could not be relied on. The corrected branch is
covered by regression tests, but its added information cost is not justified
by the current task outcome.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_counterfactual_calibration.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_counterfactual_outcome32.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_release020_calibration.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_velocity_residual_calibration.json`
- `le-wm-main/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/task_aligned_residual_phase_calibration.json`

### Adaptive task-aligned verification: drift repair and frozen confirmation — 2026-10-21

The q=0.25 Cube task probe failed for a more specific reason than a loose
threshold: it mixed early axis support with later contact effects at equal
weight, and its RGB displacement was frequently at the centroid-quantization
scale.  The new adaptive arm remains RGB-only.  It fits an affine local chart
with predeclared exponential recency `0.70`, estimates action-independent
drift, and uses that drift only for observed-versus-predicted validation; it
never enters action-induced task-progress ranking.

The q=0.25 adaptive smoke rejected the mechanism (`1/8` local releases and
`105` fallback pulses with no terminal difference).  A new, predeclared
q=0.50 sub-maximal information action was then calibrated on separate roots:
`23/40` calibration roots passed the observed-positive condition, `17/40`
abstained by sign, and the conditional threshold was `0.9683614` (rank
`22/23`).  The associated 8-root smoke is exploratory only: it increased
release to `5/8` and had local-minus-carrier mean final-distance
`-0.02318 m`, but only one non-tie win.

The smoke bundle froze all four readouts, RGB pixel map, candidate bank, and
the q=0.50 adaptive gate.  On new outcome roots `20401021--20401052`, the
runner recorded `calibration_frozen=true` and loaded the same gate without any
refit.  The paired result is `-0.01621 m` local minus matched adaptive carrier,
with bootstrap 95% CI `[-0.03711, -0.00058]`, `5/27/0`
wins/ties/losses, `23/32` released roots, and `135` fallback pulses.  The
exact two-sided sign test is `p=0.0625`, so this is a directional independent
benefit with no observed loss—not yet a statistically certified stable gain.
The controller and immutable bundle must remain unchanged for the next
independent replication; q, decay, and threshold are not tunable from this
outcome.

Artifacts:

- `le-wm-main/results/gdd_wm_cube_adaptive_task_probe_20261021_smoke8_cuda/closed_loop.json` (q=0.25 mechanism rejection)
- `le-wm-main/results/gdd_wm_cube_adaptive_task_probe_q50_20261021_smoke8_cuda/closed_loop.json` (q=0.50 bundle-producing smoke)
- `le-wm-main/results/gdd_wm_cube_adaptive_task_probe_q50_20261021_confirmation32_cuda/closed_loop.json` (immutable-bundle confirmation)

### Locked-policy replication boundary — 2026-10-21

The second 32-root panel used the *identical* q=0.50 controller and immutable
bundle, changing only outcome roots to `20411021--20411052`.  It kept similar
local-release coverage (`22/32`) but did **not** reproduce the first panel's
terminal gain: local minus carrier mean was `-0.00148 m`, bootstrap 95% CI
`[-0.00444, 0]`, `1/31/0` wins/ties/losses, exact sign `p=1.0`.  The first
panel's result must therefore not be pooled post hoc or presented as a stable
controller advantage.

The paired task decomposition localizes the remaining boundary.  In the first
panel, all meaningful benefit arose on task `3` (`3/6` wins, mean `-0.08337 m`);
the replication retained only one such win (`1/6`, mean `-0.00790 m`).  Tasks
`1`, `4`, and `5` often released the local chart and changed many actions but
still had exactly zero terminal difference.  This is not a gate-coverage
failure: it is an outer reachability/phase failure in which a verified local
XY effect is insufficient to alter the task outcome.  The next repair must
therefore condition local release on an RGB-observable task-progress/reachability
state or use a phase/macro-action controller; it must not retune the verified
local chart from the two outcome panels.

Artifact:

- `le-wm-main/results/gdd_wm_cube_adaptive_task_probe_q50_20261021_replication32_cuda/closed_loop.json` (locked-policy independent replication)

### Calibration-gated teacher alignment and D2 consequence audit — 2026-10-22

The goal-conditioned predictive PointMaze control previously passed the D1
provenance/stability interface, but that interface alone did not require the
teacher to order realized task costs better than chance.  A new
calibration-only, root-bootstrap teacher-alignment gate now binds the exact
candidate dataset hash and forbids outcome-row reads.  On its 20 calibration
roots, the teacher obtained 106/179 comparable-pair agreements (`0.5922`),
with a one-sided root-bootstrap lower bound of `0.5160`; this permits a
*diagnostic* D2 projection fit, not a planning claim.

The projection runner now requires that alignment receipt and emits
identity-bound per-root/per-context outcome traces.  The new consequence audit
predeclares both a strictly negative one-sided paired upper bound and at least
10% root-level action change before calling a held-out regret reduction a
confirmed benefit.  The frozen D2 control does not satisfy that rule: only
1/20 roots (`1/60` contexts) changes its selected candidate; mean held-out
fitted-minus-baseline regret is `-0.001575`, but the upper 95% bound is `0`.
It is therefore a non-confirmation rather than a GDD success.  This separates
``faithfully distills teacher preferences`` from ``improves real decisions``
and prevents a one-root tie-break from being promoted to a repair result.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_goal_conditioned_predictive_control_20261021_cuda_identity_repaired/teacher_execution_alignment_calibration_only.json`
- `le-wm-main/results/gdd_wm_pointmaze_goal_conditioned_predictive_control_20261021_cuda_identity_repaired/d2_projection_diagnostic_alignment_gated_trace_v2/d2_projection_distillation.json`
- `le-wm-main/results/gdd_wm_pointmaze_goal_conditioned_predictive_control_20261021_cuda_identity_repaired/d2_projection_diagnostic_alignment_gated_trace_v2/heldout_execution_consequence_audit_v2.json`

### Counterfactual contrast mechanism audit — 2026-10-22

The preceding D2 non-confirmation could have had three incompatible causes:
the frozen predictor might erase candidate-conditioned effects, the predicted
effects might not track realized dynamics, or a preserved latent effect might
be decoded incorrectly into goal-relative endpoint geometry.  A new
post-execution, per-root/per-context audit separates these possibilities using
only already materialized terminal states and costs.  It neither refits the
teacher nor changes a policy.

On the same held-out 20-root PointMaze panel (60 candidate contexts), the
dominant labels were 3 predicted-effect collapses, 17 latent-dynamics
mismatches, 12 endpoint-readout geometry mismatches, 8 task-score readout
mismatches, and 20 groups without a dominant label under the frozen
thresholds.  The key split is that mean pairwise correlation from frozen
predicted latent variation to realized endpoint variation was `0.4033`, while
the current frozen latent-to-XY endpoint readout achieved only `0.1088`.
Thus some apparent "world-model failures" are actually task-chart failures;
they must not be repaired by indiscriminate dynamics retraining.  The next
prospective repair is therefore a separately frozen, goal-conditioned endpoint
readout intervention, restricted to calibration-supported groups; local
verified dynamics remains the appropriate repair only for the 17 genuine
latent-dynamics mismatch groups.

This is mechanism evidence, not a new teacher validation or planning gain.
The artifact is
`le-wm-main/results/gdd_wm_pointmaze_goal_conditioned_predictive_control_20261021_cuda_identity_repaired/teacher_counterfactual_contrast_mechanism_audit_v2.json`;
the reusable audit is
`le-wm-main/experiments/gdd_wm/audit_teacher_counterfactual_contrast.py`.

### Prospective repairability transport contract — 2026-10-22

The contrast audit labels cannot be used directly as a controller: they are
post-execution labels.  A root-disjoint pilot now learns a coarse repair-family
router only from train roots, freezes its confidence threshold on one half of
calibration roots, certifies it on the other half, and treats outcome labels as
evaluation only.  It additionally requires both sub-mechanisms of each repair
family to appear in calibration: endpoint-readout and task-score mismatch for
metric repair, and collapse plus latent-dynamics mismatch for dynamics repair.

The existing shared-anchor teacher package fails this contract for a concrete
reason.  Its train, calibration, and outcome action-hash sets have zero
overlap (`0/5` for train/calibration and `0/5` for train/outcome).  The
resulting train-only decision-feature support gate accepts `0%` of calibration
and outcome contexts, so the pilot correctly rejects with
`rejected_train_feature_support_transport` and all deployment routes abstain.
The prior, ungated calibration authorization was therefore invalid as a
prospective repairability claim and is superseded by this fail-closed result.

This does not show that prospective repairability is impossible.  It shows
that the old package was built to test held-action generalization, not to
certify repair-family routing under common candidate support.  The fresh V7
panel now records held/train action-hash overlap in its immutable static
preflight and requires `shared_candidate_in_support`; it is the appropriate
GPU-only successor for this question.  No V7 efficacy claim exists until its
CUDA job and independent outcome audit complete.

The V7 runner now revalidates this contract at execution time, not just in its
launcher: context metadata must name `shared_candidate_in_support` and the
goal-balanced interaction design; every held action hash must occur in the
source training bank; and the context must bind a readable passed V7 preflight
ledger whose held/train overlap equals the executing held bank.  This prevents
a manual runner call from silently reverting to the old held-action protocol.

A further execution audit found that subset support alone was insufficient:
the former V7 builder would fit cross-fitted scorer/router labels on a
64-action train bank but calibrate and evaluate regret on a 16-action held
bank.  Because regret and score margins are candidate-set dependent, this is
still a transport shift even when all 16 actions are individually supported.
V7 now uses the same 16 held actions for scorer training, calibration, and
outcome, and requires `held_exact_shared_bank` at preflight, build, and runner
boundaries.  The V7 static preflight passed with 16/16 action overlap and
the exact-bank contract.  This repair removes an experimental confound; it is
not an efficacy result.

### V7 candidate-bank decision-headroom contract — 2026-10-22

Exact action support alone is insufficient for a meaningful action-ranking
test.  A fixed bank can have diverse oracle winners while its best and
second-best actions are nearly tied, leaving no measurable downstream reward
for a correct predictor to recover.  The goal-balanced V7 design now records
root-wise terminal-goal oracle headroom on its 64 disjoint candidate-design
roots before creating train, calibration, or outcome examples.  The immutable
ledger rejects a bank unless all of the following hold: normalized winner
entropy is at least `0.50`; q10 best-versus-second-best task-cost gap is at
least `0.001`; at least half of design roots have a gap of at least `0.003`;
and q10 best-to-worst candidate-cost spread is at least `0.05`.

The immutable V7 ledger passed: entropy `0.8616`, 15 distinct winners, q10 top-2
gap `0.001005`, decision-sensitive-root fraction `0.546875`, and q10
candidate-cost spread `0.05279`.  It additionally binds the ordered-action
router to 95 declared pre-outcome features, 128 train roots, and a 16-root
minimum surplus.  These values are candidate-design evidence only: they do
not access outcome roots and do not establish a learned-model, router, or
planning gain.  They make a subsequent null V7 result interpretable as a
model/repair failure rather than an action-bank near-tie or p>=n artifact.

The GPU-only V7 launcher binds
`results/pointmaze_planning_shaped_advantage_v7_static_preflight_v7_alias_contract_20261022.json`.
The ledger also freezes the executable router interface: `advantage_ridge`,
the `full_ordered_20x2` action representation, and the
`train_root_only_four_fold_blocked_advantage_mse` regularization-selection
rule.  The runner revalidates those fields and the realized 95-dimensional
router feature vector, so a manual invocation cannot silently replace V7 with
the legacy summary/logistic path.  This is an execution-integrity repair, not
an efficacy result.

V7 also materializes the C1-to-C2 mechanism link after execution: it performs
a root-disjoint representation-radius/realized-contrast aliasing audit, then
passes that immutable audit to the scorer-disagreement decomposition.  The
revised immutable ledger binds `source_history_model`, radius quantile `0.25`,
contrast gap `0.05`, and all 64 outcome-root pairs before CUDA construction.
This tests whether empirical history aliasing overlaps roots whose selected
candidate changes.  It cannot select a router, reopen calibration, or support
causal attribution.

The preceding V5 CUDA mechanism audit also identifies the correct target for
the V7 router change.  On its 32 outcome roots, the planning-shaped scorer
selected a different candidate on 7 roots and lowered candidate-bank regret on
5 of those 7; the old binary router released only 2 of those 5 gains.  Its
separate calibration certificate failed, so fallback to the cost scorer was
the only lawful deployment policy.  This is not evidence that the repair arm
is generally useful, but it rejects the simpler explanation that there was no
repairable disagreement at all.

The corresponding root-disjoint decision-aliasing replay does not support
history aliasing as the dominant explanation for that V5 router failure: 4/32
outcome roots were empirical near aliases, with no exact representation
collision, and only one of those four overlapped the seven scorer-disagreement
roots.  The alias audit is a post-execution local diagnostic, not causal
attribution, but it makes the more direct mechanism test clear: V7 must test
whether a continuous prospective regret advantage can recover the available
planning-shaped improvements without leaking calibration or outcome labels.

V7 therefore predicts the continuous cross-fitted `cost_regret -
planning_regret` advantage rather than a binary win label.  Because the
ordered-action router has 95 pre-outcome features for 128 train roots, its
ridge penalty is now selected over a fixed grid by a separate four-fold,
root-blocked train-only validation.  The selected penalty, each blocked MSE,
MAE, sign accuracy and predicted-positive coverage are serialized in the
report.  Calibration and outcome labels remain unavailable during this
selection, and calibration still independently freezes and certifies the
release threshold.  This removes a fixed-regularization variance confound; it
does not create a V7 efficacy result.

Artifacts:

- `le-wm-main/results/gdd_wm_pointmaze_goal_conditioned_predictive_control_20261021_cuda_identity_repaired/prospective_repairability_routing_pilot_v4.json`
- `le-wm-main/experiments/gdd_wm/audit_prospective_repairability_routing.py`
- `le-wm-main/results/pointmaze_planning_shaped_advantage_v7_static_preflight_v7_alias_contract_20261022.json`

### V7 shared-bank advantage-router result — 2026-10-22

The V7 CUDA panel completed on the RTX 5060 Ti after the static support and
headroom contracts were frozen.  It uses 128 train, 192 calibration, and 64
root-disjoint outcome contexts; all roles score the same 16 supported ordered
action sequences.  The full 95-dimensional visual-history/action router is
fit only on cross-fitted train-root advantages, and its ridge penalty is
selected only by blocked train-root validation.

The candidate-bank and raw-router diagnostics are nontrivial:

| quantity | outcome result |
| --- | ---: |
| cost-only mean regret | `0.01814` |
| planning-shaped mean regret | `0.02090` |
| frozen raw-router mean regret | `0.01407` |
| raw-router release coverage | `8 / 64` |
| released roots with strict improvement | `5 / 8` |
| scorer-disagreement roots | `30 / 64` |

This is **not** a deployment gain.  The independent calibration certificate
rejected the router: its upper 95% bound for routed-minus-incumbent regret was
positive (`0.00582`).  The authorized outcome policy therefore fell back to
the calibration incumbent (`planning`), whose mean outcome regret was
`0.02090`, worse than cost-only.  The raw router's outcome value is retained
as a post-freeze diagnostic, not a policy-selection result.

The completed aliasing audit finds 10/64 near decision-alias witnesses and
only 5 overlap with the 30 scorer-disagreement roots.  Thus history aliasing
is neither a necessary nor sufficient explanation for the routing failures.
The mechanism audit also records 11 missed effective improvements among 16
improvable disagreement roots.  The concrete remaining failure is therefore
calibration transport of the continuous advantage score, not fixed-bank
headroom, action support, or an unexercised planning arm.

During completion, a post-execution audit initially treated V7 as V9 and
required `routing_eligible == candidate_disagreement`.  That assertion is
wrong for V7: it deliberately scores every root, while same-choice roots have
zero decision opportunity.  The audit now keys that check to the frozen
`opportunity_gated` contract; regression coverage accepts V7 all-root
eligibility and continues to reject a V9 same-choice release.  This repair
only restores correct audit semantics; it does not alter the immutable GPU
report or retroactively authorize its raw-router result.

Artifacts:

- `le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/report.json`
- `le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/planning_shaped_repairability_mechanism_audit.json`
- `le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/decision_aliasing_calibration_outcome.json`
- `le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/cost_planning_disagreement_audit.json`
