# PointMaze C1: balanced reversal result and position-alias boundary

## Scope

This artifact records two sealed C1 experiments. Neither is a C2
repairability result, C3 authorization result, closed-loop planning result,
or a claim about universal world models.

## V12: balanced reversal does not yet yield stable history value

V11 allowed sign-mixed pairs with highly dominant winners. V12 instead used
fresh roots and required both winners to occupy at least 25% of each of two
disjoint design folds. Its static ledger passed with eight summary-colliding
pairs, a pairwise q10 cost gap of `0.01189`, and 97.85% margin coverage.

The frozen calibration split selected a recurrent visual-history effect model
and permitted it over the action-effect-only baseline: the calibration
root-cluster sign gain was +2.34 percentage points (95% bootstrap lower bound
+0.17 points). The held-out roots did not confirm a reliable benefit:

| held-out, root-clustered metric | recurrent minus action-effect-only |
| --- | ---: |
| pairwise sign accuracy | +0.82 pp, 95% CI [-0.41, +2.07] pp |
| contrast MAE improvement | -0.222 pp, 95% CI [-0.296, -0.145] pp |

Thus the held-out sign direction is positive but not authorized, and the
recurrent effect model is reliably worse calibrated in contrast magnitude.
The result rejects the claim that balanced realised winners alone stably
extract transferable decision history.

The V12 score's `action-only` arm is an action-effect-only model, not a
task-blind action-ID baseline: both arms use the same source position and
goal in the cost readout. On V12 outcomes, a fixed per-pair majority-winner
rule has only a 56.8% ceiling, while the action-effect-only model reaches
76.3%. The remaining baseline strength is therefore not explained by winner
imbalance; task geometry plus generic action effects carries real information.

## V13: current action family has no position-alias headroom

V13 predeclared a stronger static admission rule on a new root extension. In
addition to contrast, coverage, and 40% per-fold winner balance, each action
pair had to defeat a cross-fold proxy that uses only source position, target
goal, and the other fold's mean terminal displacement for each action. The
proxy deliberately omits visual history and velocity.

The immutable V13 preflight was rejected before loading an encoder or opening
any calibration/outcome labels:

```text
proposal count:                         1536
contrast/coverage/balance admissible:     90
position-only proxy admissible:            3
intersection:                              0
required selected pairs:                   8
```

This is an action-bank/environment boundary, not a reason to weaken the
threshold. The present 20-step random pulse-pair family can create balanced
task preferences, but it cannot simultaneously make those preferences
unpredictable from position/goal plus a global action-effect chart.

## Consequence

The next C1 test must change the controlled mechanism, not increase GRU width
or tune the selector after V12. The appropriate successor is a
velocity-matched-history PointMaze state pool: paired visual histories should
arrive at nearly matched source positions with opposed source velocities, then
evaluate braking/acceleration candidate pairs. Its static preflight must
again require contrast, two-fold balance, and position-proxy rejection before
any learned fit. If that construction cannot produce eight pairs, PointMaze
should be reported as unable to instantiate the desired C1 mechanism under
the declared visual/action interface, and the test should move to a contact
domain rather than relaxing the gate.

## V14/V15: the mechanism, not the model capacity, had to change

The first velocity-matched construction (V14) used random source positions
and a twenty-step inward-versus-outward constant acceleration pair. Its static
admission found no action pair satisfying both a true opposed-velocity winner
reversal and rejection of the position/goal/global-effect proxy. This is an
informative negative construction result: with a fixed target, the direction
that moves toward the target remains mostly identifiable from source position,
so opposed incoming velocity did not create the intended decision alias.

V15 therefore uses a fresh, goal-centred radial state pool. Each pair arrives
at the same position on a radius-0.28 ring around the declared target while
its velocities point toward versus away from the target. Candidate proposals
now compare weak and strong corrections along the same direction, for which
the appropriate magnitude can depend on the incoming velocity. Its immutable
static ledger passed before any encoder/model fit:

| static property | V15 value |
| --- | ---: |
| pairwise absolute cost-gap q10 | 0.09342 |
| nontrivial contrast coverage | 1.000 |
| minimum two-fold opposed-velocity winner reversal | 0.8125 |
| maximum cross-fold position-only proxy accuracy | 0.53125 |

The V15 ledger therefore establishes a controlled C1 *mechanism* with roots,
candidate actions, and design folds isolated before learned calibration or
held-out outcomes. It does **not** yet establish a history-model gain,
repairability, planning improvement, or universal-world-model result. Those
claims remain conditional on the separately running host-GPU train,
calibration, and outcome evaluation.

## V17: renderer-visible high-velocity histories yield a held-out C1 gain

V15's first GPU build was invalidated before fitting because its declared
causal history could include a marker outside the renderer-visible region.
V16 repaired that contract but reduced the source speed so much that only one
of eight candidate pairs passed static admission. V17 keeps V15's
high-velocity radial mechanism while requiring every causal history marker to
remain within `[0.05, 3.95]` in both position dimensions. The new pool has
832 fresh roots (`24712`--`25543`), source speed `1.5`, and passed all 16
causal renderer checks per root before any encoder was loaded.

Its immutable static ledger passed all declared C1 gates:

| static property | V17 value |
| --- | ---: |
| pairwise task-contrast q10 | `0.33533` |
| contrast coverage | `1.000` |
| minimum two-fold opposed-velocity winner reversal | `1.000` |
| maximum position-only proxy accuracy | `0.500` |
| causal history renderer visibility | passed |

The host-CUDA run used 256 train, 256 calibration, and 256 root-disjoint
outcome roots. The candidate actions are deliberately shared across these
splits: this is a history-identifiability test under matched action support,
not an action-OOD generalization claim. The recurrent history model was chosen
using only one calibration-root half and authorized on the other. At that
authorization stage, its pairwise sign-accuracy gain over action-only was
`+0.500` with root-cluster bootstrap 95% CI `[+0.418, +0.587]`.

The sealed outcome evaluation reproduces the effect on fresh roots:

| held-out, root-clustered metric | recurrent vs. action-only |
| --- | ---: |
| pairwise sign accuracy | `1.000` vs. `0.500` |
| sign-accuracy gain | `+0.500`, 95% CI `[+0.442, +0.558]` |
| contrast MAE | `0.00834` vs. `0.36495` |
| contrast-MAE reduction | `0.35660`, 95% CI `[0.34359, 0.36917]` |

As a secondary consequence analysis, the same frozen model was then evaluated
on every candidate in the shared 16-action bank. This analysis uses a separate
read-only root-cluster audit and does not change V17 model selection:

| held-out fixed-bank metric | recurrent minus action-only |
| --- | ---: |
| mean realized top-1 regret | `0.00271` vs. `0.20648` |
| regret reduction | `0.20377`, 95% CI `[0.18228, 0.22482]` |
| top-1-correct roots | `151 / 256` vs. `32 / 256` |
| top-1-accuracy gain | `0.46484`, 95% CI `[0.40625, 0.52734]` |

The outcome interval is a separate read-only audit which reloads the
frozen model and resamples roots rather than individual pairs. Outcome labels
were materialized only after the frozen-selection hash existed. The result
therefore supports the narrow C1 statement that, in this renderer-visible,
velocity-aliased PointMaze mechanism, a visual history model extracts action
preference information that source position, goal, and action effects alone
do not recover.

This is not yet evidence that the repair is prospectively selectable, improves
full-bank planning, transports to contact dynamics, or supports online visual
control. The next valid step is C2: use pre-outcome diagnostics to decide when
to apply a history-conditioned repair and test that frozen decision on a new,
root-disjoint outcome panel. The V17 preflight presently reuses the V16
generic schema name for compatibility with the shared builder; the immutable
ledger is valid for this C1 run because its dataset, root range, seeds,
candidate design, and visibility contract are separately pinned. A successor
must receive a native V17 schema rather than relabel this sealed artifact.

Artifacts:

- `le-wm-main/results/pointmaze_pairwise_contrast_v17_visible_high_velocity_static_preflight_20261023.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v17_visible_high_velocity_20261023_cuda/temporal_ood_v6_frozen_selection.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v17_visible_high_velocity_20261023_cuda/score/temporal_ood_v6_score_report.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v17_visible_high_velocity_20261023_cuda/v17_outcome_root_cluster_audit.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v17_visible_high_velocity_20261023_cuda/v17_full_bank_outcome_root_cluster_audit.json`

## V18: a pre-fixed policy independently confirms the C1 result

V18 is not another selection attempt. Before its pool, calibration labels, or
outcome labels were opened, it fixed the V17 recurrent configuration
(`hidden=32`, contrast weight `1.0`, temperature `1.0`, and
`source_history_model`). It then constructed a separate 832-root pool using a
new pool seed, new action-bank seeds, and root identifiers `25544`--`26375`.
Its native V18 ledger verified the same renderer-visibility, pair completeness,
velocity reversal, position-proxy rejection, root isolation, and action-support
requirements before host-CUDA execution.

The fixed policy cleared a fresh calibration-only authorization gate: pairwise
sign gain over action-only was `+0.19483` with a bootstrap lower bound
`+0.17362`. Outcome labels were then opened only after the frozen-selection
receipt was written. The separate, read-only outcome audit confirms the
matched-pair effect on 256 fresh outcome roots:

| independent held-out, root-clustered metric | recurrent vs. action-only |
| --- | ---: |
| pairwise sign accuracy | `1.000` vs. `0.500` |
| sign-accuracy gain | `+0.500`, 95% CI `[+0.44189, +0.55811]` |
| contrast MAE | `0.00813` vs. `0.35839` |
| contrast-MAE reduction | `0.35026`, 95% CI `[0.33794, 0.36304]` |

The independently audited full 16-action bank produces the same directional
decision consequence without changing the policy:

| independent held-out fixed-bank metric | recurrent minus action-only |
| --- | ---: |
| mean realized top-1 regret | `0.00366` vs. `0.19860` |
| regret reduction | `0.19494`, 95% CI `[0.17400, 0.21595]` |
| top-1-accuracy gain | `+0.54688`, 95% CI `[+0.48438, +0.60938]` |

V17 and V18 therefore establish a reproducible *C1 mechanism result*: in the
declared renderer-visible velocity-alias PointMaze construction, visual history
adds held-out fixed-bank action-ranking information beyond source position,
goal, and action-only effects. The two panels do **not** establish prospective
repairability (C2), online control authorization (C3), contact-domain
transfer, or a universal world-model property. The next experiment must test a
pre-outcome rule deciding when that history repair should be used across
mechanisms, rather than retuning the recurrent model on this confirmed
mechanism.

V18 artifacts:

- `le-wm-main/results/pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_static_preflight_20261023.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda/temporal_ood_v6_frozen_selection.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda/score/temporal_ood_v6_score_report.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda/v18_outcome_root_cluster_audit.json`
- `le-wm-main/results/pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda/v18_full_bank_outcome_root_cluster_audit.json`
