# PointMaze prospective repairability matrix V1

Status: implementation and synthetic-contract tests complete; no fresh-root
outcome has been opened under this protocol.

## Question

Can a route chosen before outcome execution identify where a history repair is
preferable to the action-effect carrier, beyond four simple alternatives:
visible motion, predictive MSE, declared action support, and ATM?

## Frozen comparison

Every outcome root receives scores from the same frozen backbone, shared
in-support candidate bank, horizon, and source-supervision budget:

| Policy | Root-level choice |
| --- | --- |
| action-effect-only | matched action-conditioned carrier |
| uniform-history repair | history-conditioned effect model on every root |
| ordinary preference | task-cost preference ranker with the same source budget |
| repairability diagnostic | pre-outcome diagnostic route |
| visible motion / MSE / support / ATM | four separately frozen simple-route controls |

The primary estimand is the root-paired true-regret reduction of the diagnostic
route relative to uniform history repair.  Its required secondary comparisons
are action-effect-only, ordinary preference, and each simple-route control.
Root-level bootstrap intervals and sign-flip tests are reported; the seven
predeclared comparisons receive Holm correction.

## Seal contract

`decision.json` contains only candidate scores and root routes.  It must state
`outcome_labels_opened: false`, contain no task costs or realised terminal
states, and is hashed into the separately generated `outcomes.json`.  The
audit rejects any leakage, missing route, candidate mismatch, or outcome hash
mismatch.  Thus neither audit nor post-outcome analysis can choose a route or
threshold.

The protocol is an offline fixed-bank transfer test.  It is not a PlanWAM
reproduction, conditional safety guarantee, online MPC result, contact-domain
claim, or universal world-model certificate.
