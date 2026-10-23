# Planning-shaped counterfactual control: current audit

Status: `mechanism evidence; not a general performance claim`.

This audit treats a planning-shaped future/effect objective as a strong baseline,
not as a reproduction of PlanWAM.  All three arms receive the same visual
history, goal, candidate-action summary, MLP capacity, optimization budget, and
initialization-seed ensemble:

- `cost_regression`: terminal task-cost supervision;
- `ranking_augmented`: cost plus within-root ranking supervision;
- `planning_shaped`: cost plus ranking plus realized terminal-effect supervision.

The central question is whether the last objective consistently reduces real,
candidate-set regret on unseen roots.  It does not ask whether PlanWAM's driving
system succeeds or fails.

## Protocol corrections made before the fresh panels

The original diagnostic implementation assigned a different initialization seed
to each arm.  That confounded supervision with optimizer randomness and its
output is not used for objective comparisons.  Version `v0.3` makes every arm
use the same initialization-seed ensemble.  Version `v0.4` additionally writes
paired per-root decisions.  The matched-panel builder now accepts a declared
root range and constrains both evaluation and interaction-design roots to that
range; otherwise a new dataset extension could accidentally sample old roots.

The relevant implementation is
[`run_pointmaze_planning_shaped_control.py`](../le-wm-main/experiments/gdd_wm/run_pointmaze_planning_shaped_control.py)
and the panel builder is
[`build_pointmaze_temporal_ood_v6_matched_panel.py`](../le-wm-main/experiments/decision_benchmark/build_pointmaze_temporal_ood_v6_matched_panel.py).

## Fresh, root- and action-disjoint evidence

Every run used a four-seed shared-initialization ensemble, 64 train roots, 48
calibration roots, 32 outcome roots, a frozen visual-history/goal/action input
contract, and GPU preflight.  Outcome candidates were executed only after
calibration selection.  V1 and V2 use state-interaction candidate banks; V3
uses a predeclared goal-balanced variant selected only with disjoint design-root
terminal effects and the declared goal.

| Panel | Candidate bank | Calibration-selected arm | Outcome cost regret | Outcome planning-shaped regret | Planning minus cost |
| --- | --- | --- | ---: | ---: | ---: |
| V1 | state interaction | cost | 0.02697 | 0.03008 | +0.00311 |
| V2 | state interaction | planning-shaped | 0.05656 | 0.04339 | -0.01317 |
| V3 | goal-balanced interaction | planning-shaped | 0.02555 | 0.03440 | +0.00885 |

Thus planning-shaped training produces a local gain in V2, but that gain does
not transfer consistently: it loses to cost-only in V1 and V3.  The result is
not evidence that planning-shaped representations are useless; it is evidence
that the present calibration signal cannot reliably identify when the additional
future-effect objective will improve counterfactual candidate ranking.

## Mechanism identified so far

Paired outcome decisions reveal limited effective contrast.  Cost-only and
planning-shaped select the same candidate in 26/32 V1 roots, 22/32 V2 roots,
and 23/32 V3 roots.  Consequently, a few disagreement roots determine the
mean-regret direction.  The goal-balanced bank improved *design-root* oracle
diversity (normalized entropy 0.848, 15 distinct winners), but did not make the
two learned arms reliably disagree or transfer their calibration preference to
outcome roots.

For the planning-minus-cost calibration regret difference, paired bootstrap
95% intervals crossed zero in every panel.  A calibrated selector therefore
has insufficient panel-level evidence to authorize the more complex arm under a
fail-closed protocol.  This rules out interpreting the V2 positive direction as
a deployable gain, while preserving it as an informative heterogeneous case.

The V5 report is now also processed by the reusable post-execution
arm-disagreement audit.  Cost-only and planning-shaped select the same
candidate on 25/32 roots.  All nonzero decision difference is concentrated in
the remaining 7 roots: planning-shaped improves 5 and worsens 2, for a
within-disagreement mean planning-minus-cost regret of `-0.07422` (full-panel
mean `-0.01624`).  This is a mechanism decomposition, not a selective policy
claim: V5's router still failed independent certification.  The machine-readable
artifact is
[`cost_planning_disagreement_audit.json`](../le-wm-main/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/cost_planning_disagreement_audit.json).

## Consequence for the research program

The next object is not another planning-shaped loss.  It is a prospective,
root-level repairability decision:

```text
pre-outcome history/action/support features
        -> predict whether a metric/effect repair can lower regret
        -> repair only when supported; otherwise probe or retain the safe arm
```

## First prospective routing control and protocol repair

V4 is the first fresh panel that trained a root-level visual-history/action
router on four-fold cross-fitted train-root labels.  It did not read raw source
state in its router features.  Its calibration-selected threshold released the
planning-shaped scorer for 5/32 outcome roots and achieved 0.80 release
precision.  This was better than cost-only, but not better than unconditional
planning-shaped scoring:

| V4 outcome policy | Candidate regret | Planning release coverage |
| --- | ---: | ---: |
| Cost-only | 0.07518 | 0.000 |
| Planning-shaped | 0.06540 | 1.000 |
| Original routed policy | 0.06777 | 0.156 |

The original router searched and evaluated its release threshold on the same
48-root calibration split.  Although the selected threshold lowered
calibration regret from 0.05006 (unconditional planning-shaped) to 0.04456,
it failed to transport: it withheld planning-shaped scoring on four outcome
roots where it would have helped.  V4 is therefore a **diagnostic result**, not
evidence that prospective repairability routing is validated.

Version `v0.2` of
[`run_pointmaze_planning_shaped_repairability.py`](../le-wm-main/experiments/gdd_wm/run_pointmaze_planning_shaped_repairability.py)
repairs this selection bias.  It hashes calibration roots into disjoint roles:

```text
selector calibration roots       -> select the fixed incumbent and threshold
certification calibration roots  -> one-sided paired bootstrap certificate
                                      against that incumbent
outcome roots                    -> execute only the authorized policy
```

The routing policy is authorized only when its independent certification-root
upper 95% paired regret difference is non-positive and its observed mean
improvement is strictly negative.  Otherwise it falls back to the selected
fixed arm.  Unit tests cover both meaningful authorization and required
fallback, alongside the raw-state isolation contract.

The router also now receives a fixed four-statistic geometry of the two arms'
*predicted* terminal effects: between-arm disagreement, each arm's
within-candidate spread, and the separation of their selected candidate
effects.  These are decision-time model outputs only; realized terminal states,
raw simulator state, and outcome labels remain forbidden.  This repairs the
previous information mismatch in which a future-effect objective was evaluated
by a router that saw only cost margins.

A retrospective V4 diagnostic supports the necessity of this repair but is
not independent evidence for it: the selector chose planning-shaped as the
fixed incumbent, while certification yielded an upper 95% paired regret delta
of `+0.00645`.  The repaired policy would therefore reject the router and
deploy planning-shaped for every V4 outcome root, matching its lower observed
fixed-arm regret (`0.06540`).  This calculation reads an already materialized
V4 outcome and cannot be used to validate the revised protocol.

## V5: fresh prospective test and recovered mechanism artifact

V5 is the independent test of the repaired protocol.  It used fresh roots
9336--9655, the state-interaction matched-support bank, the same four-member
shared-initialization ensemble, and a recorded CUDA preflight (RTX 5060 Ti,
CUDA 13.0).  The builder reports no overlap with historical roots or action
hashes; held pulses are exact members of the declared train-action support.

The selector side of calibration chose cost-only as the fixed incumbent.  On
the disjoint certification side, the routed-minus-cost mean regret difference
was `+0.00390`, with one-sided paired-bootstrap upper 95% bound `+0.01806`.
The release rule therefore correctly rejected routing and froze **cost-only**
as the only authorized deployment policy.

| V5 root-disjoint outcome report | Candidate regret | Planning release coverage |
| --- | ---: | ---: |
| Cost-only | 0.04886 | 0.000 |
| Planning-shaped (descriptive fixed arm) | 0.03263 | 1.000 |
| Raw router (descriptive; not authorized) | 0.04545 | 0.094 |
| Authorized deployment (certificate fallback to cost-only) | 0.04886 | 0.000 |

This is not a validated router win.  It is a sharper failure mode: the
planning-shaped arm has an independent positive direction on this outcome
panel, but the available calibration certificate cannot safely select it.
Conversely, the unapproved raw router improves over cost-only but remains
worse than unconditional planning-shaped.  The supported conclusion is that
small-sample root-level calibration is presently the limiting step, not that
the planning-shaped objective has no decision value.

### Post-execution aliasing audit

The original V5 report was produced by a legacy serializer that omitted the
already-computed `post_execution_outcome_labels` field.  The original report
is preserved unchanged.  A migration tool replays only the frozen deterministic
outcome context after the completed report exists, writes a separately hashed
label sidecar, and is accepted by the audit only through an explicit
`--outcome-labels` argument.  This is a reproducibility repair, not a new
training run or a selection input.

Using 48 calibration roots to fix the representation-neighborhood radius and
32 root-disjoint outcome roots to query it, the visual-history audit found
4/32 nearby history pairs (12.5%) whose realized candidate-cost contrast gap
was at least 0.05.  There were no exact representation collisions.  Thus V5
supplies empirical local decision-aliasing evidence, but no global
identifiability lower bound or deployment certificate.

The artifacts are:

- [`report.json`](../le-wm-main/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/report.json)
- [`post_execution_outcome_labels_recovered.json`](../le-wm-main/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/post_execution_outcome_labels_recovered.json)
- [`decision_aliasing_calibration_outcome.json`](../le-wm-main/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/decision_aliasing_calibration_outcome.json)

The next fresh panel must predeclare a changed structural repair or
calibration design before outcome generation; V5 cannot be retuned after this
observation.  V6 therefore does not alter V5's router threshold: it replaces
the detached auxiliary-effect readout with a terminal-endpoint-coupled score
and expands the independent disagreement stratum.  Its protocol is
[`POINTMAZE_COUNTERFACTUAL_ENDPOINT_V6_PROTOCOL_20261021.md`](POINTMAZE_COUNTERFACTUAL_ENDPOINT_V6_PROTOCOL_20261021.md).
