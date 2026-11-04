# PointMaze performance line: freeze boundary

This line studies planning performance, not the stopped non-immediate
mechanism. The 2026-11-03 budget-allocation run is sealed as an entire
optimizer/scheduler version: both allocation and CEM standard-deviation
definition changed. The existing four development roots and two budget levels
are four paired roots, not eight independent observations.

The sealed run has spectral, matched-random, nominal and feedback-CEM arms;
15 physical actions; a cp10 real-RGB refresh; root prediction-token caps of
900 and 1800; and a further 96 cp10 prediction tokens per arm. It retains
actual call counts and times and reports `full_cost_certified=false`. A
prediction-token cap does not by itself equalize encoder, finite-difference,
SVD, CPU/GPU time or total planner cost. The measured development results are
not sufficient to pick a budget after looking at outcomes, assert a cost
advantage, or freeze a confirmation method.

Before any independent performance confirmation, preserve the two budget
points or predeclare a new policy without conditioning on confirmation
outcomes; bind the full optimizer, candidate generation, scores, seed policy,
feedback, action clock and hashes; and enforce instrumentation from before
candidate generation through feedback replanning. Specify how all arms can
use their permitted resources, record encoding, prediction, differential and
SVD work and actual wall time separately, and explain shared costs. A common
scalar budget may be used only with an explicit, defensible accounting model;
otherwise report resource curves and separate operation counts.

Independence requires separate evidence for root generation and complete
historical exclusion, normalizer fit membership, and checkpoint training
membership. The existing P1 prerequisite audit is a useful fail-closed
reference, but it cannot certify a new performance manifest by assertion.
Current normalizer and checkpoint provenance limitations are documented in
`PC_POINTMAZE_NORMALIZER_CONFIRMATION_BOUNDARY_20261103.md` and
`D2P_P1_INDEPENDENT_CONFIRMATION_PREREQ_AUDIT_20261102_EXCLUSIVE.md`.
Until authoritative sources are available, provenance is unknown and no
independent performance confirmation is authorized by this record.

At confirmation time, predeclare the unit of analysis, sample size, eligible
root generation, failures, paired effect estimates and interval/multiple
comparison rule; compare spectral against nominal, matched random, and
feedback CEM, not CEM alone. Keep predictive performance separate from PC2
mechanism and the historical P1 gate.
