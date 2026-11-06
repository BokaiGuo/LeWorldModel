# Full-action suffix finite-difference baseline: E0 development protocol

Compare the existing full-action CEM with the bounded finite-difference
score-gradient optimizer on four new exact-Wall tasks: starts
`(-0.5, y, theta)`, with `y ∈ {-0.21, 0.21}` and
`theta ∈ {-1.12, 1.12}`, goal `(0.5, 0.19)`. Reject overlap with the
explicitly checked panels. These are privileged witness-prefix
diagnostics, not planner, PC2 or PC3 performance tests.

Use the existing fixed-height geometric constructor once per root;
choose its first legal witness scoring `<0.05` or record the task as
ineligible, without replacement. Freeze the first five witness actions.
Create three shared initial tails: zero, plus two independent uniform
tails in `[-0.5, 0.5]` from a preregistered root-specific seed. Evaluate
the three initial chains once (nine shared prediction tokens). Both
optimizers use exactly the same initial tails and initial scores. For
each start, each method may use up to 120 full-chain evaluations. CEM
uses its existing full-action configuration; finite difference uses
`0.05` perturbations and line steps `(0.1, 0.25, 0.5, 1.0)`. Report
actual calls and unused budget: finite difference may stop early or leave
an insufficient remainder and must not consume dummy work.

Report success `<0.05`, best scores, failures, counts and paired
quality-versus-cost. Audit all actions and scores with alternate
same-environment geometry, verify common starts and that selected
incumbents were scored. Stop after four roots, with no replacement,
extra starts or tuning. This is a development baseline, not evidence
that either optimizer is globally sufficient or that source histories
are independent.
