# v20 actual-candidate trust filtering: exact-Wall development

Post-hoc v19 trace analysis found large finite-step linearization errors
on actual generated candidates. v20 tests a new deterministic trust rule:
after each candidate is scored nonlinearly, compare its actual feature
change with the nominal finite-difference prediction. Allow it into
intermediate retention only if absolute error is at most
`0.05 + 0.5 * ||actual change||`. Nominal candidates remain available;
every scored complete chain, including a filtered one, remains eligible
as a final incumbent. This is a new development hypothesis, not a
post-hoc change to the v19 experiment. The combined feature is position
and scale-1 task score, and these thresholds are not calibrated guarantees.

Compare v17, v19, v20 and four-start CEM on eight new exact-Wall tasks:
start `[-0.5,y,theta]` with `y` in `{-0.18,0.18}` and `theta` in
`{-1.03,1.03}`; goal `[0.5,gy]` with `gy` in `{-0.30,0.30}`. Check
against enumerated previous panels, but broader source independence is
unknown. Use the shared global bank, 15-step horizon, same 4,095-token
cap and final nonlinear score. Charge every scored sample and successor
probe. Record trust admissions, retained candidates, selected chains and
cost, then audit saved actions/endpoints against alternate same-environment
geometry.

Report terminal distance and success `<0.05` for all attempted tasks.
Stop at eight tasks, one attempt per arm; no retries, substitutions,
threshold adjustments or expansion. This panel is exploratory and cannot
establish PC2, independent PC3, or full hardware-cost equivalence.
