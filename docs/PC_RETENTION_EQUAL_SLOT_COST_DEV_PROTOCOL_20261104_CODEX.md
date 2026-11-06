# Equal-slot retention and full-cost development protocol

Date: 2026-11-04

This is a new development panel for retention-policy attribution. It uses a
task family not present in the prior named panels and compares four retention
policies on the same response-generated candidate construction:

- `response`
- `greedy`
- `diverse`
- `random`

Each policy has the same beam width, greedy-slot count, model, normalizer,
action domain, horizon, random seed rule, and total prediction-token cap. The
global shared-CEM arm is included as a separate strong comparison. Every arm
pays its own response, successor-probe, selection, refinement, and final
comparison costs. No posthoc suffix score, privileged witness, or physical
outcome enters selection.

The primary development quantities are complete-chain terminal score, success
under the predeclared `<0.05` threshold, abstention, prediction tokens,
prediction calls, successor probes, and SVD calls. The task is a paired
development unit; the 16 roots are not treated as independent confirmation
episodes.

This panel can identify a reproducible retention-policy signal or provide a
bounded negative result. It cannot by itself establish PC2, PC3, real RGB
control, source independence, or generalization.
