# P1 fixed-lattice common-tail candidate protocol

## Motivation

The recent constrained common-tail model run evaluated 20 non-neutral points
per exposed root and produced no eligible row.  Root `20263160` had no point
passing the absolute-tail conjunction; root `20263161` had some points passing
the adjusted-tail conjunction but none passing both absolute and adjusted
conditions.  The minimum positive constraint loss therefore does not establish
feasibility.  This protocol tests broader fixed directions without using those
results as an action oracle.

## Fixed generation and selection

Before model evaluation, the bank is fixed as:

1. the 20 positive/negative coordinate directions in the 10-dimensional
   parameter chart;
2. 20 deterministic balanced sign directions from a fixed Walsh-style sign
   lattice;
3. one radius, `0.10`, for every direction;
4. one neutral row, followed by the 40 directions in the listed order.

The selector is exactly the existing common-tail conjunction:
immediate cost must be at least neutral plus `1e-6`; mean final delta must be
at most `-1e-6`; every absolute final delta must be non-positive; mean
neutral-adjusted tail delta must be at most `-1e-6`; and every adjusted delta
must be non-positive.  Among eligible rows, the minimum sum of final deltas
wins with proposal-index tie breaking.  If none is eligible, the result is
`None`; there is no fallback.

The evaluator receives only one parameter point and returns one immediate
model cost plus five common-tail model costs.  No physical outcome, hidden
state, post-hoc success label, or privileged state is accepted by the
candidate generator or selector.

## Accounting and limitations

Nominal per-root logical cost is 41 prefix evaluations, 410 suffix-token
samples, and zero gradient VJPs.  Encoder, goal-Q, device synchronization,
memory, wall time, and model loading remain uncharged and must be accounted
separately.  This is not an equal-budget comparison with the prior searches.

The protocol is model-only and CPU-tested with analytic callbacks only.  It
does not establish that the lattice contains an eligible point, that the model
predicts physical controllability, or that P1 is admitted.  The 40-point bank
is a fixed development proposal, not a result and not a reason to alter the
gate.

No CUDA launcher, physical executor, or historical result is modified by this
change.  Any future CUDA adapter must authenticate the same checkpoint, source,
goal-Q, root, action chart, frozen parameters, and protocol hash before a run;
GPU use requires separate coordination.
