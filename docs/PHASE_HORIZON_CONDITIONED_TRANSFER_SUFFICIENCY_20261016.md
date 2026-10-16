# Phase- and Horizon-Conditioned Transfer Sufficiency

Status: **Proposed method and confirmatory protocol.** The H-JEPA-derived
design principles in this document are not yet validated LeWorldModel results.
Measured results are called out explicitly and remain subject to their original
claim boundaries.

## Core insight

The decision value of a learned latent variable is not a static property of
that variable. It is conditional on at least four objects:

\[
\mathcal V(z)
=
\mathcal V(z\mid h,p,\mathcal A_{\mathrm{deploy}},M_{\mathrm{task}}),
\]

where \(h\) is the prediction horizon, \(p\) is the planning or contact phase,
\(\mathcal A_{\mathrm{deploy}}\) is the deployed action support, and
\(M_{\mathrm{task}}\) is the declared task metric. A feature can therefore be
useful for immediate control and harmful for long-range goal matching, or
stable over long horizons while being irrelevant to action choice.

The supplied H-JEPA research note makes this distinction concrete. Fine body
state is useful for short-horizon actuation, whereas position and traversable
structure are more useful for long-horizon navigation. Distinct temporal
levels can learn distinct representations, and high-level predictions can be
turned into reachable low-level subgoals. The important boundary is equally
useful: hierarchy helps only when the data contain identifiable time-scale
separation and enough support at each scale. Slow predictability alone is not
sufficient, because stable scene identity can still be action-irrelevant.

## Integration with the LeWorldModel evidence

This principle unifies several previously separate negative results.

1. **Memory:** fixed Fourier is a low-variance short-horizon carrier, while
   phase augmentation is useful only at selected horizons. A universal memory
   selector is therefore the wrong estimand.
2. **PushT:** mode observability can improve while within-mode angular action
   identifiability remains insufficient. A high-level phase label cannot
   replace the low-level action-effect interface.
3. **PointMaze:** support-aware fallback can improve closed-loop behavior even
   when the frozen world-model ranking remains unreliable. Long-horizon goal
   geometry and local reachable action geometry must be tested separately.
4. **Reacher:** reset roots, online replanning states, CEM proposal support,
   planner-call phase and final action execution define different decision
   strata. Pooling them produces a false stationary transfer problem.
5. **Spectral diagnostics:** spectrum, alignment and commutator are conditional
   measurements inside one declared horizon/phase/task chart; none is a
   context-free transfer certificate.

The resulting research object is not a generic hierarchical world model. It
is a **phase- and horizon-conditioned transfer-sufficiency audit and release
rule** for frozen or learned world models.

## Proposed model

Let \(\mathcal H_t\) denote causal observation/action history. For each
declared level \(\ell\), learn or fit a representation

\[
z_t^{(\ell)}=\Phi_{h_\ell,p_t}(\mathcal H_t),
\qquad h_1 < h_2 < \cdots < h_L.
\]

The low-level representation is evaluated on local action effect and successor
consistency. A higher-level representation is evaluated on long-horizon task
progress and subgoal reachability. It is not accepted merely because its
prediction loss is low. Its release gate is

\[
\Gamma_{h,p}
=
\Gamma_{\mathrm{support}}
\land \Gamma_{\mathrm{mode}}
\land \Gamma_{\mathrm{action}}
\land \Gamma_{\mathrm{transfer}}
\land \Gamma_{\mathrm{subgoal}}.
\]

The five terms require, respectively:

- deployment actions and states to lie inside calibrated support;
- the relevant mode or planning phase to be observable;
- within-mode action effects to be identifiable;
- the representation to add held-out transfer information beyond action-only,
  MSE and the lower-level representation;
- a proposed high-level subgoal to be reachable by the next lower level.

If any gate fails, control falls back to the lower level or abstains. It must
not silently reuse a representation or risk threshold calibrated for another
horizon, phase or proposal distribution.

## Action relevance prevents a slow-feature shortcut

Long-horizon prediction naturally suppresses fast variables, but it can retain
slow nuisance variables. Every higher-level representation must therefore pass
an action relevance control:

\[
\Delta R_{h,p}
=
R(\text{action-only/lower-level})
-
R(\text{candidate higher-level representation}),
\]

with matched no-dynamics and shuffled-prediction controls. A representation is
not promoted when shuffled predicted dynamics perform as well, or when the
gain is explained by source-action interaction alone. This is the direct link
between H-JEPA-style selective abstraction, ATM action identifiability, HTSC
release and GDD-WM decision geometry.

## Frozen-backbone intervention before hierarchical retraining

The first experiment must not immediately train a full hierarchical backbone.
On the same frozen encoder, root split and candidate bank, fit separate
train-only projections for \(h\in\{1,2,4,8,16\}\) and declared phases. Compare:

1. one shared latent and one shared goal cost;
2. horizon-specific projection with the same flat planner;
3. horizon-specific projection plus action-identifiability supervision;
4. top-down subgoal planning using the accepted higher-level projection;
5. no-dynamics, shuffled-prediction and privileged reachability ceilings.

This order distinguishes representation hierarchy from temporal decomposition.
Only if a horizon-specific projection adds held-out value should a multi-level
encoder/predictor be trained.

## Required measurements

Each `(environment, architecture, seed, horizon, phase)` stratum must report:

- prediction MSE and multi-step closure error;
- task-progress monotonicity along expert or successful paths;
- action inverse risk and within-mode action-ranking regret;
- realized/predicted successor mismatch;
- proposal-support coverage and fail-closed rate;
- lower-level subgoal reachability;
- closed-loop regret and task success;
- no-dynamics and shuffled-prediction controls;
- paired uncertainty and the number of independent roots.

The primary estimand is the held-out regret change conditional on the lower
level and action-only baseline, not probing accuracy or reconstruction quality.

## Success and failure criteria

A hierarchical transfer claim requires all of the following on untouched
roots and at least one unseen environment or architecture:

- the higher-level representation improves long-horizon task progress or
  ranking beyond the lower-level and action-only baselines;
- the improvement survives shuffled-prediction and no-dynamics controls;
- the lower level can reach the emitted subgoals under deployment support;
- closed-loop regret improves without degrading local safety or contact;
- the same calibration rule is used without test-outcome tuning.

Failing any item leaves a useful boundary result. In particular, lack of a
fast/slow data gap, insufficient long-horizon samples, unreachable subgoals,
or a source-action readout that dominates predicted dynamics are explicit
no-go conditions for hierarchical retraining.

## Current Reacher anchor — measured, 2026-10-16

The frozen protocol was rerun on the deployment-aligned 128-root Reacher panel
using host CUDA, with 64 train, 32 calibration and 32 untouched outcome roots.
The fixed source-action MLP reduced outcome regret from `0.5319` to `0.2226`;
the paired delta was `-0.3093`, with 95% CI `[-0.4311,-0.1930]`. Adding the
predicted-dynamics features increased regret to `0.3116`; shuffled prediction
reached `0.1919`, better than the unshuffled full model. Thus this panel
contains a reproducible phase-specific source-action decision signal but no
independent evidence that the current predicted latent dynamics add transfer
information.

This result motivates phase/horizon conditioning; it does not validate H-JEPA,
causal latent physics, a universal world model, or stable online planning.
The artifact is
`le-wm-main/results/gdd_wm_cross_environment_20261016/reacher_deployment_aligned_panel128_capacity_frozen/capacity_audit.json`.

The first fresh online test did not convert the absolute-action result into a
stable gain (`31/64 -> 33/64`) and worsened final qpos error. Its reported
`4.30%` support statistic was later invalidated because inactive planner rows
were included in the denominator; the corresponding empirical-quantile gate
was also too strict for finite candidate clouds. This correction narrows the
cause from generic support absence to coordinate transport across CEM proposal
clouds.

The repaired `proposal_relative` chart standardizes actions within each cloud,
while a calibration-only one-sided Wilson lower-bound gate freezes the
finite-sample threshold at `0.927105`. Offline source-action regret improves
from `0.47152` to `0.21495`, but full predicted dynamics (`0.25498`) still lose
to shuffled prediction (`0.19389`). Two root-disjoint fresh online batches
improve success `32/64 -> 38/64` and `36/64 -> 39/64`; the batch-stratified
aggregate is `68/128 -> 77/128`, paired `+0.0703125`, 95% CI
`[+0.015625,+0.1328125]`. Conditional active-row support averages `0.93933`
with no abstention. Final and minimum qpos-distance intervals still cross zero.

This establishes a scoped decision-repair consequence: phase-conditioned
source-action information can improve fresh Reacher binary success when its
action chart is proposal-relative and support-calibrated. It does not establish
independent predicted-dynamics value, continuous-error improvement, or a
universal hierarchy. The remaining transfer-sufficiency bottleneck is exactly
the no-dynamics/shuffled-prediction gate.

Two explicit attempts to cross that gate failed on fresh roots. A fixed `0.5`
fusion with frozen predicted-to-goal cost improved final distance but changed
success `39/64 -> 34/64` and did not beat a within-cloud shuffled-cost placebo.
A calibration-only top-k protocol then asked model cost to tie-break only among
source-action elites. No `k` from `2` to `64` beat shuffled cost with a CI below
zero, so the preregistered behavior is `k=1` (no-dynamics fallback). An
exploratory fresh `k=2` panel gave true and shuffled the same `39/64` success.
Thus goal conditioning alone does not restore candidate-paired action-effect
information; the next admissible intervention must alter predictor training or
decision-geometry distillation, not tune another runtime mixing weight.
