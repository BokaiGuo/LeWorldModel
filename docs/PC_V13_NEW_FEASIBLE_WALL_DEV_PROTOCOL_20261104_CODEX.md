# New Wall plan-then-feasibility development protocol

The recent suffix diagnostics show that their finite searches rarely cross
the wall, but their tasks were not certified feasible. On a separate new
four-task panel, first run v13 and a same-global-bank full-action CEM
control, then seal and audit plans before constructing any privileged
feasibility witness. The witness may not enter either planner.

Tasks are the Cartesian product of starts `[-0.5,y,theta]`,
`y ∈ {-0.26,0.26}`, `theta ∈ {-1.2,1.2}`, with goal `[0.5,0.19]`.
Check task equality against the named previous v13/v14 prospective and
suffix-diagnostic panels and the older panels referenced by the prospective
runner. Broader history remains unknown. One run per arm and task, no
replacement, retry, threshold adjustment, or tuning.

Both arms share v13's fixed global proposal bank, horizon 15, action bounds
[-1,1], same goal and 1920 five-step token cap. v13 uses fixed seed
`2029161000 + root`. The control scores that same bank, then uses remaining
budget for full-action (15-knot) CEM with seed `2029162000 + root`, and
performs the same first-five-zero paired comparison. Record every requested
float32 action, position, score, full-chain selection and meter. Save plans
before any witness.

Audit all plan actions using the alternate same-environment geometry,
identical banks, token totals, and that selected chains were scored full-chain
minima. Only after a successful plan audit, compute the existing privileged
two-sided geometric witnesses once per root, convert to legal normalized
float32 actions, and check both original and alternate geometry. Save both
successes and failures, steps, actions, and trajectories. A 15-step witness
within 0.05 demonstrates existence only; failure to find one leaves
feasibility unknown. The method has no access to the witness.

Stop after four paired plans and two geometric witness attempts per task.
This is a small exposed development diagnostic, not PC2 intervention, PC3
confirmation, or an independent checkpoint/source claim. Equal prediction
tokens do not equalize all response/SVD/search computation.
