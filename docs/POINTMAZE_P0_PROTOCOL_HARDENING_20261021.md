# PointMaze P0 protocol hardening

## Why the earlier evidence could not attribute a failure mechanism

An oracle ladder is only meaningful if every arm evaluates the same held roots,
the same ordered candidate actions, the same horizon, task objective, and
execution/replanning semantics.  Earlier artifacts recorded descriptive
`cohort_id` and `candidate_bank_id` fields, but those strings did not bind the
actual root set or candidate action tensors.  Two nominally matched arms could
therefore differ because they used different starts or different actions.

The prior PointMaze candidate-task contrast audit had a separate issue: it
claimed that its task readout was train-only without checking whether the
readout export contained calibration or outcome roots.  It also compared a
squared predicted goal distance with an Euclidean realized task cost.  The
first can leak evaluation roots into a decoder; the second changes contrast
margins and invalidates calibration-scale interpretation even when rankings
are unchanged.

## Executable changes

`experiments/audit_planner_oracle_separation.py` is now schema v2.  A P0
specification and every supplied arm must bind:

- ordered outcome-root identities and their SHA-256 hash;
- ordered candidate action-sequence hashes and their SHA-256 hash;
- candidate count and a shared-fixed-bank declaration;
- horizon, task objective, and execution semantics.

Any missing binding, altered action order, altered root set, or mismatched
execution semantics is fail-closed as `protocol_mismatch`.  The audit only
routes completed, genuinely matched oracle arms; it does not select a model
using test outcomes.

`experiments/decision_benchmark/audit_candidate_task_contrast.py` now rejects
a fit export whose `episode_id` intersects a candidate dataset calibration or
outcome root.  Predicted and realized cost are both Euclidean goal distance.

## Consequence for existing artifacts

Earlier planner/oracle reports lacking the v2 identity bindings are historical
diagnostics, not P0 mechanism evidence.  They cannot support claims such as
"ranking-limited" or "dynamics-limited."  A fresh P0 panel must produce all
four predeclared oracle arms under a shared v2 contract:

1. ground-truth candidate-bank oracle;
2. perfect-dynamics MPC;
3. real-successor with learned task geometry;
4. learned dynamics with oracle task readout.

Only then may the audit route a failure to candidate/horizon, ranking,
information/support, or deployment/planner integration.  The candidate oracle
remains diagnostic only and must never enter RGB-only action selection.

## Candidate-context repairability replication

The execution-grounded repairability router was also run on the two previously
unexecuted shared-action contexts (`01` and `02`) with the checkpoint, root
roles, seed, selector grid, and deferred-outcome protocol unchanged.  These
are new candidate interventions on the same 100 recorded roots, not independent
root replications.

| Candidate context | PSD regret | MLP regret | Routed regret | Router coverage | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `00` (prior) | 0.02407 | 0.03035 | 0.02407 | 0.00 | not confirmed |
| `01` (new GPU run) | 0.02055 | 0.04952 | 0.02055 | 0.00 | not confirmed |
| `02` (new GPU run) | 0.04005 | 0.04028 | 0.04005 | 0.00 | not confirmed |

All three contexts passed the deferred-outcome/four-way-split contract audit.
The transport aggregate deliberately makes no pooled confidence claim because
the contexts reuse roots.  Its useful conclusion is negative but specific:
the present pre-outcome repairability router does not transport across these
candidate contexts; its abstention is not an isolated threshold, GPU, or
single-bank implementation failure.  This remains a privileged state/action
selector diagnosis, not a result about visual GDD distillation or deployment.

## Validation

The regression suite includes a same-name/different-action-bank counterexample
and a fit-export/outcome-root leakage counterexample.  Both must fail closed.
