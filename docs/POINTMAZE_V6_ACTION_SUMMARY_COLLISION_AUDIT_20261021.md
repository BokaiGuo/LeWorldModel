# PointMaze V6.7 action-summary collision audit

Status: `completed static audit; not a new learned-model result`.

## Question

The historical PointMaze V6.7 candidate bank was described as an
equal-energy pulse-order interaction bank.  That property alone does not show
that the historical action encoder lost the relevant temporal information: the
encoder also retained the first and final action, which a random permutation
can change.

This audit asks a narrower, executable question: within each candidate bank,
are there distinct 20-step action sequences that collide under the historical
`mean/std/first/last/length` summary?

## Evidence

The audit read only candidate action sequences from:

`le-wm-main/results/pointmaze_temporal_ood_v6_7_state_interaction_20261021_cuda/temporal_ood_v6_context.json`

It inspected 768 calibration rows (48 roots x 16 candidates) and 512 outcome
rows (32 roots x 16 candidates).  A pair required distinct full sequences and
summary equality up to absolute tolerance `1e-12`.

Result: all 80 roots had zero qualifying summary-collision pairs.  The audit
therefore rejects the strict collision contract even when the required minimum
is only one pair per root.

## Interpretation

This does **not** invalidate V6.7 as an equal-energy, pulse-order interaction
experiment.  It does establish that V6.7 cannot identify whether a negative
result arose because the historical summary discarded interior action order:
its random permutations usually changed the first or final pulse, leaving a
legal discriminative signal in the old representation.

Accordingly, V6.7 must not be cited as evidence for or against the narrower
action-summary aliasing mechanism.

## Forward protocol

The unexecuted counterfactual-endpoint V6 panel now uses pairs with identical
mean, standard deviation, first pulse, final pulse, energy, and length, while
permuting only the three interior pulses.  Its runner refuses to fit unless
every calibration and outcome root has at least eight such distinct pairs.
It also reports a same-panel factorial diagnostic for summary versus ordered
actions and cost-head versus endpoint readout.  No outcome from that new
protocol exists yet.
