# PointMaze V6: counterfactual endpoint-coupled repair protocol

Status: `predeclared; unexecuted`.

## Why V6 exists

V5 found a real but non-deployable positive direction for the legacy
planning-shaped arm.  Its independent calibration certificate selected
cost-only even though the outcome planning-shaped regret was lower.  The
mechanism audit found that 25/32 outcome roots selected the same candidate
under both scorers, so all decision changes were concentrated in seven
disagreement roots.

The code inspection identified a second, structural limitation.  The legacy
arm trains terminal-effect prediction as an auxiliary head, but deployment
ranks candidates with a separately optimized cost head.  Consequently future
effect supervision is not structurally required to determine the selected
action.

V6 tests a different, narrow hypothesis:

> If a visual-history/action model predicts candidate-conditioned terminal
> states and candidate score is read directly as distance between that endpoint
> and the declared goal, then terminal supervision may produce a more stable
> decision repair than an auxiliary future-effect loss.

This is not a PlanWAM reproduction, a claim that endpoint prediction is
universally counterfactually sufficient, or an online RGB planning result.

## Frozen design

Inputs are exactly `source_history_model`, declared goal, and the complete
ordered `20 x 2` candidate action pulse.  Raw simulator source state remains a
training target/protocol audit field only and is forbidden at the endpoint
arm's deployed input.  V1--V5 retain their historical mean/std/first/last
action summary; V6 instead preserves the whole sequence so state-interaction
pairs that differ only in interior pulse order cannot be collapsed by the
feature contract.

The state-interaction bank enforces that collision by construction: paired
candidates have identical action mean, standard deviation, first pulse, last
pulse, total energy, and length, while a non-identity permutation changes only
the three interior four-step pulses. This is an action-representation stress
test, not a change to the action amplitude envelope or support regime. The
runner independently verifies at least eight distinct summary-collision pairs
for every train, calibration, and outcome root before any model is fitted.

For the V6 summary-control arms, the legacy summary is reduced after a stable
lexicographic action-row sort. This removes order-dependent floating-point
summation residue (about machine precision) without adding action information,
so a declared collision is an exact model-input collision. Historical V1--V5
artifacts retain their original, non-canonical encoder unchanged.

After fitting but before outcome terminal states are materialized, the runner
also evaluates every collision pair through both canonical-summary controls.
Their cost and endpoint scores must agree within `1e-12`; otherwise the panel
rejects an unintended downstream action-identity side channel.

This requirement is prospective: the historical V6.7 interaction bank failed
the same collision audit on every evaluated root, so its result is not reused
as evidence for this narrower representation mechanism. See
[`POINTMAZE_V6_ACTION_SUMMARY_COLLISION_AUDIT_20261021.md`](POINTMAZE_V6_ACTION_SUMMARY_COLLISION_AUDIT_20261021.md).

Before fitting, the V6 runner fail-closes unless train, calibration, and
outcome root identities are pairwise disjoint, the context declares
`shared_candidate_in_support`, and outcome rows contain no realized terminal
state, task cost, or terminal reference.  This makes deferred outcome
execution an executable contract rather than a builder convention.

The same-panel primary comparison and factorial diagnostics are:

| Arm | Candidate score | Role |
| --- | --- | --- |
| Canonical-summary cost-only | Direct predicted cost head | Ordered-action stress control |
| Cost-only | Direct predicted cost head | Primary ordered-action incumbent |
| Planning-shaped auxiliary | Direct cost head with effect/ranking auxiliary loss | Descriptive V5 bridge |
| Canonical-summary endpoint | `|| predicted_terminal_xy - goal ||` | Ordered-action stress control |
| Counterfactual endpoint | `|| predicted_terminal_xy - goal ||` | Primary ordered-action repair |

The primary authorization comparison is still ordered cost-only versus ordered
endpoint, so it holds the complete ordered action representation fixed while
testing endpoint-coupled readout.  The two summary arms are descriptive,
same-panel controls that expose the representation contribution and its
interaction with readout.  They are not eligible for post-calibration policy
selection.  This prevents a positive ordered endpoint result from being
misread as a cross-panel V5-to-V6 comparison, while retaining the legacy V5
planning-shaped arm as a historical bridge.

For the endpoint arm, the loss is terminal-state prediction plus task-readout
and within-root ranking terms.  Its score is always recomputed from the mean
predicted endpoint, never a separately trained cost head.

The panel uses fresh state/action roots `9656..10295`, generated with seed
`202610341`, and a fixed goal-balanced interaction candidate bank with held
actions in source-training support.  Sixty-four disjoint design roots may use
the already-declared goal and simulator only to select a high-diversity oracle
winner bank; they never overlap train, calibration, or outcome roots. This
avoids testing a representation repair on a bank with a near-global default
winner. The builder serializes the design-root winner entropy and the runner
requires it to be at least `0.50` before GPU fitting. It has 128 train, 192
calibration, 64 outcome, and 64 disjoint interaction-design roots. The CUDA launcher is
[`run_pointmaze_counterfactual_endpoint_v6_cuda.sh`](../le-wm-main/experiments/gdd_wm/run_pointmaze_counterfactual_endpoint_v6_cuda.sh).

Before any CUDA build the launcher runs a static preflight over the exact prior
context ledger, root range, candidate support, strict collision count, and
goal-balanced winner diversity. It then binds the build to that ledger's
absolute context paths and SHA-256 values: any added, removed, or byte-changed
eligible historical context rejects the launch rather than silently changing
the exclusion set. The panel metadata and post-run contract audit retain the
ledger SHA-256 and declared root counts. The frozen current ledger passed this check:
33 contexts excluded, `128/192/64` split roots, 64 design roots, 8 canonical
summary-collision pairs, winner entropy `0.861`, and dominant-winner fraction
`0.203`. The artifact is
[`pointmaze_counterfactual_endpoint_v6_static_preflight_20261021.json`](../le-wm-main/results/pointmaze_counterfactual_endpoint_v6_static_preflight_20261021.json).
This is readiness evidence only, not a model fit or outcome result.

After the runner completes, a separate fail-closed contract auditor checks the
saved report before any alias or disagreement analysis is emitted. It verifies
that the primary policy compared only ordered cost versus ordered endpoint,
that every root met the strict collision support threshold, and that both
canonical-summary controls were score-aliased before outcome execution. A
contract-audit pass establishes protocol eligibility only; it does not certify
endpoint efficacy.

## Selection and certification

V6 deliberately does **not** tune a root router after seeing V5.  It fixes a
single endpoint policy and observes whether it differs from cost-only.  The
calibration roots are deterministically hash-split into selector,
arm-certification, and independent contrast-risk-calibration roles.
The frozen V6 root seed and current prior-context exclusion ledger yield 73,
63, and 56 roots respectively in these roles; this was checked by the static
preflight before any V6 model fitting or outcome materialization.

1. The selector can nominate the endpoint arm only if it has lower mean regret
   than cost-only and at least 12 model-disagreement roots.
2. The arm-certification role independently requires at least 12 disagreement
   roots, strictly negative endpoint-minus-cost mean regret, and a non-positive
   one-sided paired-bootstrap 95% upper bound.
3. The third role calibrates a finite-sample, root-level maximum task-contrast
   envelope for the ultimately authorized arm at `alpha=0.1`.  A root is
   permitted only when its predicted winner is separated from every fixed-bank
   competitor by that frozen envelope.
4. Otherwise deployment is fixed cost-only.  Outcome terminal states are
   materialized only after arm selection, arm certification, and the contrast
   envelope are frozen.

The minimum count is a support gate, not a power claim.  It prevents a result
like V5, where only four certification disagreement roots determined whether a
nontrivial repair could run.

## Primary outcomes and failure interpretation

Primary estimand: held-out mean candidate regret of the **authorized** policy.
Report additionally includes endpoint-minus-cost regret on pre-outcome
model-disagreement roots, ranking disagreement coverage, descriptive legacy
planning-shaped regret, a fixed-bank regret--coverage certificate for the
authorized arm, and a post-execution calibration-to-outcome history aliasing
audit.  The contrast certificate is marginal over exchangeable roots and this
declared bank; it is not a state-conditional or adaptive-CEM guarantee.

The report additionally emits a descriptive 2-by-2 table for canonical-summary
versus ordered action representation and cost-head versus endpoint readout, including
the two main contrasts and their difference-in-differences interaction. These
diagnostics never enter arm authorization.

The V6 report also records the spread of the four initialization replicas for
the endpoint arm before outcome terminals are materialized: member agreement
with the frozen mean-endpoint choice, the number of distinct member winners,
and selected-endpoint XY spread. This is deliberately diagnostic-only. It
cannot tune a threshold, alter an action score, or authorize a V6 policy; it
only distinguishes a future uncertainty-gating hypothesis from the existing
detached-readout and action-aliasing hypotheses.

| V6 outcome | Interpretation |
| --- | --- |
| Endpoint certificate passes and held-out authorized regret improves | Evidence for this fixed PointMaze endpoint-coupled repair under declared support |
| Endpoint descriptive arm improves but certificate rejects | Repair may help on average, but finite-sample selection/certification remains unresolved |
| Endpoint does not improve despite adequate disagreement support | V5 failure was not caused solely by the detached auxiliary readout |
| Disagreement support is insufficient | Candidate bank/representation does not expose enough testable repair events; no efficacy claim |
| Endpoint improves only in the unapproved legacy bridge | No endpoint-repair claim; inspect decision aliasing and candidate geometry before a new protocol |

V6 must not be retuned after outcomes.  Any subsequent change to score readout,
candidate design, support count, model width, or certificate is a new panel
with a new root range.
