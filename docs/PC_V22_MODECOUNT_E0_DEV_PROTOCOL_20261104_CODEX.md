# v22 mode-count paired exact-Wall development protocol

Hypothesis: the single response mode in v22 can restrict candidate coverage;
allowing two target-ranked modes may improve complete-chain quality under the
same total budget. It can also spend resources that would otherwise go to
refinement or successor probes. Neither direction is assumed.

Use eight development tasks with starts `(-0.5, y, theta)`, `y=±0.21`,
`theta=±1.07`, goals `(0.5, gy)`, `gy=±0.18`. Reject exact overlaps with
the explicitly enumerated earlier panels; broader provenance remains unknown.
Use the same v22 compiler and full 1,920 prediction-token cap for each arm,
changing only `max_modes=1` versus `max_modes=2`; include the same-bank
multistart CEM with the same cap. Pair tasks and initial bank, fix seeds in
the runner, and retain all failures in the denominator.

Record all actual proposals, candidate counts per depth, successor probes,
refinement generations, selected actions, costs and runtime. Audit every
saved action/score with alternate implementation of the same geometry and
check first-bank equality, selection and actual ledger. Report terminal
distance and the existing success threshold `<0.05`. Stop after these eight
tasks; do not replace or tune roots. This tests a budget-allocation tradeoff,
not a pure directional effect, PC2 mechanism, independently sourced PC3
advantage, or the high-dimensional spectral contract.
