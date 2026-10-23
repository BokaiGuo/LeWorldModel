# PointMaze shared-future planning control

## Status

**Static preflight complete; no fresh GPU fit was authorized.**  The original
V8 interval had a root-capacity configuration error and was repaired as V8-r2.
V8-r2 then passed root isolation, action support, candidate diversity, cost
spread, and the explicit capacity reservation, but failed the unchanged
lower-tail headroom gate: q10 best-versus-second oracle gap was `0.000334`,
below `0.001`.  Thus no panel, model fit, or outcome labels were materialized.
This document reports neither a planning result nor a PlanWAM reproduction.

## Why this control exists

The prior `planning_shaped` PointMaze arm is action-conditioned: its trunk
receives visual history, goal, and the candidate action.  It is a valid
counterfactual control, but it is not the same architecture as a
shared-future representation followed by candidate scoring.  Treating the two
as synonymous would make a comparison with planning-shaped-future work
ambiguous.

`experiments/gdd_wm/run_pointmaze_planning_shaped_control.py` now accepts
`--include-shared-future-control`.  The added arm has this information flow:

```text
visual history + goal -> shared future context -> shared-effect head
candidate action + shared future context -> candidate score
```

The shared context never reads a candidate action.  Its auxiliary future
target is the within-root mean executed terminal effect, so the future target
is identical across the root's candidate actions.  The score head may read the
declared action, as any candidate evaluator can.  This is a deliberately
small, falsifiable PlanWAM-inspired control; it is not the original PlanWAM
architecture or a claim about that method's empirical performance.

## Frozen comparison requirements

The shared-future arm may be evaluated only where the following are identical
across cost-only, action-conditioned planning-shaped, and shared-future arms:

1. train/calibration/outcome root split;
2. ordered candidate action bank and action representation;
3. model width, optimization steps, initialization seed ensemble, and compute
   budget;
4. calibration-only objective selection;
5. deferred outcome execution and post-selection scoring of every frozen arm.

The main outcomes are candidate regret, ranking accuracy, zero-regret rate,
and regret--coverage after independent HTSC calibration.  Effect RMSE is
reported as a diagnostic only: the shared-effect target is root-level by
construction and is not intended to stand for per-action counterfactual
prediction quality.

After the three arms have completed, the V8 launcher runs
`evaluate_pointmaze_shared_future_v8_htsc.py`.  It reads the frozen
per-candidate tables and uses only calibration roots to fit (i) a simultaneous
fixed-bank contrast envelope and (ii) a model-only score-ambiguity threshold.
It then reports, separately for direct cost regression, action-conditioned
planning-shaped, and shared-future arms:

- release count and coverage on 64 independent outcome roots;
- wrong-release rate, measured only after the already-frozen selection;
- realized utility and regret relative to the direct-cost fallback;
- exact root, action-bank, report, and static-ledger provenance.

The certificate is marginal over the declared fixed candidate bank.  It is
not a permit-conditional or state-conditional safety claim, and it does not
authorize adaptive candidate generation such as CEM.

## Isolation rule

Existing PointMaze outcome panels cannot be reused to validate this new arm:
their outcomes have already informed prior analysis and method development.
The arm requires a newly extended dataset and an immutable root/action-bank
ledger created before any encoder fit, scorer training, calibration, or
outcome execution.  The current V7 panel remains reserved for its frozen
advantage-router protocol, so it must not be augmented post hoc with this
arm.

`experiments/gdd_wm/run_pointmaze_shared_future_v8_cuda.sh` is the sole
prospective V8-r2 entrypoint. The original `10808..11319` interval became
infeasible before fitting because later immutable context ledgers consumed
some of its 512 roots: it needed 384 evaluation roots plus 64 disjoint
interaction-design roots. No model fit or outcome was produced from that
failed preflight. V8-r2 validates CUDA first, creates the independent
`17720..18359` root extension (640 roots, leaving explicit capacity slack)
only after that gate, freezes a V8 ledger, then
builds the matched panel and runs all arms with `full_ordered_20x2`. It
refuses CPU fallback and refuses an existing panel or outcome path.

The control runner independently reopens the ledger at its own execution
boundary.  It refuses to fit unless the receipt bytes, root identities,
root counts, action sequences, shared-in-support bank, deferred-outcome rule,
and ordered-action interface all still agree with the frozen V8 contract.
This prevents an accidental direct invocation with a different panel from
being misreported as a scientific failure of the shared-future control.

This separation tests the research question fairly:

> Does an action-independent planning-shaped future context add decision value
> beyond a direct action-conditioned scorer, under the same counterfactual
> support and evaluation protocol?

## V8-r2 eligibility result

The host-CUDA launcher passed its device and recurrent-cuDNN preflight, then
created the independent `17720..18359` extension.  The repaired root-budget
check reported 640 eligible roots, 448 required for the 384 evaluation plus 64
interaction-design roots, and 192 reserved surplus roots.  This confirms that
the earlier failure was a configuration defect, not a GPU or data-access
failure.

The immutable static ledger nevertheless rejected the candidate family before
model fitting because the legacy-summary collision bank had insufficient
tail decision headroom.  Its decision-sensitive-root fraction was `0.6875`
and its q10 cost spread was `0.13550`, but the q10 best/second gap was only
`0.000334`.  The artifact is
[`pointmaze_shared_future_v8r2_static_preflight_20261022.json`](../le-wm-main/results/pointmaze_shared_future_v8r2_static_preflight_20261022.json).

This is evidence about feasibility of this collision-stress candidate family,
not evidence for or against a shared-future predictor.  The threshold is kept
fixed: relaxing it after seeing the new roots would turn near-tied oracle
choices into an apparent learned-ranking test.
