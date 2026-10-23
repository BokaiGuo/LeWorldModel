# TwoRoom action-effect replication audit

The reports labelled `20261021`, `v2`, `v3`, and `final` are not independent
replications: all contain the same 16 held-out roots.  Their effective
independent outcome-panel count is one.  In addition, the original/v2 pair and
the v3/final pair share byte-identical `panel_arrays.npz` files.

The new audit artifact is
`le-wm-main/results/tworoom_action_effect_repairability_20261021_replication_independence_audit.json`.
It rejects the panel family as nonindependent and prevents those directories
from being pooled as four experiments.

`run_tworoom_action_effect_v4_panels_cuda.sh` predeclares three new disjoint
160-root panels.  Its primary estimand is the full effect adapter's paired
closed-loop terminal-distance gain over action-only and no-dynamics controls,
with held-out candidate-ranking regret as a coequal secondary estimand, not
merely its gain over frozen LeWM.  This is a P1 replication test; it does
not claim prospective repairability until a separate pre-outcome router is
validated.

## Inference contract hardening

The V4 aggregate now treats the three panels, not the 96 pooled outcome roots,
as the independent replication units. For each paired terminal-distance,
success, and ranking-regret gain it reports the three panel means, an equal-panel
estimate, a panel-level t interval and exact two-sided sign-flip test, a nested
panel/root bootstrap interval, and a root bootstrap explicitly conditional on
the observed panels. With only three panels, the panel-level test is coarse;
this audit cannot establish broad environment generality.

The auditor also verifies that train, calibration, and outcome roots are
disjoint within each panel and that no root appears in any role of another
panel. Thus a reused root cannot be hidden by moving it between split roles.
These checks are implementation/protocol safeguards; they do not constitute
scientific evidence until the three fresh host-CUDA panels complete.

## Fresh host-CUDA V4 result

The three predeclared panels completed on the host RTX 5060 Ti after CUDA
allocation and recurrent/cuDNN preflight.  The immutable aggregate is
`le-wm-main/results/tworoom_action_effect_repairability_v4_panels_20261023_cuda/v4_panel_audit.json`.
It covers 288 train, 96 calibration, and 96 root-disjoint outcome roots across
three independent panels.

The full effect adapter reduced final terminal distance in all three panels:
+1.495 versus action-only and +1.860 versus no-dynamics (positive means closer
to the goal).  However, it *increased* fixed-bank selected-candidate regret in
all three panels: -0.632 versus action-only and -0.533 versus no-dynamics
(positive would favour the adapter).  Success gain was -0.010 versus
action-only and 0.000 versus no-dynamics.  The exact panel sign-flip p value
is 0.25 for the distance and regret comparisons because there are only three
independent panels.

Consequently V4 does not establish the required decision-transfer gain:
shorter terminal distance without better action ranking or success is not an
adapter victory.  This is retained as the declared contact/action-effect
boundary.  The result must not be used to tune a V5 support gate, change V4
thresholds, or claim a positive TwoRoom replication.

During aggregation the audit also found and corrected a reporting defect: the
collector keeps protocol root order while the closed-loop evaluator sorts
roots.  The corrected auditor joins all paired statistics by root identity,
not array position; it does not modify any completed panel or outcome.
