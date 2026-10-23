# PointMaze V10: prospective routing to an executed endpoint repair

## Question

Can a pre-outcome diagnostic choose when an actual endpoint-coupled
counterfactual repair should replace a direct cost scorer?

V10 is deliberately a C2 intervention, not a claim that all repair families
can be selected.  The repair arm predicts a candidate-conditioned terminal
state from visual history and the full ordered 20-by-2 action sequence, then
computes the declared goal cost from that predicted terminal.  This differs
from the older repair-family audit, which predicted only a mechanism label and
did not execute a repair arm.

## Frozen protocol

The original `11832..12343` configuration was invalidated before execution:
an audit found that a valid historical context already covered roots through
`12919`. A corrected r2 static ledger used `14776..15287` but was rejected
before fitting because its goal-balanced bank failed the preregistered lower-tail
headroom gate. The r3 mean-gap robust bank also failed the same q10 gate,
although it improved the q10 value. The r4 configuration uses a new root range
`15800..16311`, appended only from r3's state-only pool, and a two-fold
tail-robust candidate-bank designer that directly maximizes the worst-fold q10
best-versus-second gap after winner diversity, without access to train,
calibration, or outcome roots. r4 still missed the hard q10 threshold by
`7.7e-06`; r5 therefore moves to a fresh `16312..16823` range and increases
only the deterministic pre-outcome proposal bank from 96 to 384 pairs. It
retains the `192/128/64`
train/calibration/outcome split.  Both cost-only and endpoint arms receive the
same frozen visual representation, root/action split, action bank, model
width, steps, and initialization budget.  Every held candidate is in the
source action support.  Interaction-rich candidate pairs are selected on
separate design roots without task-cost access.

The static ledger freezes not only the three split counts, but every exact
episode-root identity.  Panel construction and execution independently reject
a same-sized panel whose train, calibration, or outcome roots differ from that
ledger.  Candidate identity is likewise recomputed from the ordered 20-by-2
action bytes before any model is fitted.

The GPU-only launcher extends only the immediately preceding state-only pool
after recurrent CUDA allocation succeeds. It never imports a fitted router,
calibration label, or outcome artifact from an earlier V10 attempt.

Cross-fitting supplies train-root-only endpoint-minus-cost regret advantages.
The 15-dimensional router sees only scorer margins, visual-history summaries,
goal norm, and the selected actions' signed/temporal difference.  It is fit
only when cost and endpoint select different candidates.  The experiment fails
closed unless at least 31 such train roots exist.  Same-choice roots always
retain cost-only, even when the router score is high.

Calibration is split by root identity into selector and certificate roles.
Each requires at least eight disagreement roots.  Threshold selection uses
only the selector role; authorization additionally requires a one-sided paired
upper confidence bound no greater than zero against cost-only on every
certificate root.  Outcome successor labels are materialized only after that
threshold and certificate are fixed.

If either calibration role lacks its eight predeclared opportunities, V10
emits an immutable `rejected_pre_outcome_calibration_opportunity` report with
the train/calibration diagnostics and leaves every outcome successor sealed.
This is an identifiability finding, not a failed run to be repaired using the
outcome roots.

Earlier still, if cross-fitted train roots contain fewer than 31 genuine
cost/endpoint candidate disagreements, V10 emits
`rejected_pre_calibration_crossfit_opportunity`.  No final deployable scorer,
calibration-label decode, or outcome successor is then materialized.  This separates
an under-powered repairability question from a negative endpoint-repair result.

The immutable outcome audit has exactly two allowed conclusions. If a routed
policy was calibration-authorized, it is a held-out gain only when it releases
at least one root and both the paired all-root mean and one-sided 95% bootstrap
upper bound of routed-minus-cost regret are below zero. If calibration rejects
routing, the only valid conclusion is that the cost fallback was preserved.
The audit cannot tune a threshold or reopen an arm.

## Interpretation

A positive report would establish one narrow result: on this fixed PointMaze
candidate bank, a cross-fitted pre-outcome signal selected a real
endpoint-coupled repair arm with independently certified non-inferiority and a
held-out consequence.  It would not establish a generic repairability
theorem, an HTSC guarantee, or online closed-loop planning success.

Failure to meet either opportunity gate is an identifiability result.  It is
not permission to lower the gate, run CPU training, or select thresholds after
outcomes are opened.

## Static-design results and stop condition

All V10 attempts below passed the host CUDA recurrent preflight and stopped
before any panel build, model fit, calibration-label decode, or outcome
materialization when their static candidate-bank ledger failed. The sole
failed check was the preregistered lower-tail best-versus-second gap:

| Design | Fresh roots | Proposal pairs | q10 gap | Decision |
| --- | --- | ---: | ---: | --- |
| goal-balanced (r2) | `14776..15287` | 96 | 0.000374 | reject |
| mean-gap robust (r3) | `15288..15799` | 96 | 0.000639 | reject |
| tail-robust (r4) | `15800..16311` | 96 | 0.000992 | reject |
| expanded tail-robust (r5) | `16312..16823` | 384 | 0.000880 | reject |

The fixed criterion is `q10 >= 0.001`; it was never relaxed. The r4 near miss
and r5 reversal show that q10 is not stable under independent root draws, even
after aligning the design objective and quadrupling proposal coverage. This
closes the fixed 16-action endpoint-bank route for the current PointMaze
distribution. The next admissible intervention is not another proposal sweep:
it must construct a state-conditioned bank after a verified short probe, then
re-run the same root-disjoint calibration and outcome protocol on a new range.
