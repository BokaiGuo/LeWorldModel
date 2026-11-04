# v13 global proposals: fixed E0 development comparison

## Hypothesis

On a prior constructed contact goal, a privileged physical witness scored
better under the model than the chains actually scored by v12. Test the
distinct, limited hypothesis that spending part of the same budget on broad
unprivileged full-horizon actions can improve discovery. The witness is never
an input or template. This is not a claim that broad proposals constitute
program discovery or spectral advantage.

v13 first evaluates zero plus 64 random actions: 32 constant over the
horizon and 32 with independent first/second-half values. Values are iid
uniform in `[-1,1]`. It passes the best complete chain into the unchanged
v12 response compiler as carry and deducts all those evaluations from the
1920-token cap. If a prior tail is present, it is scored in the same initial
batch and competes as well. The child re-scores the incumbent and applies its
ordinary recursive generation, refinement and final zero-prefix pair.

Compare four arms: v13, original v12, three-start direct CEM, and a global
proposal + full-action CEM control. The latter gets exactly the same 64 global
proposals and random seed as v13, retains the best complete initial chain,
uses remaining budget for per-step CEM and reserves a final zero-prefix pair.
It does not compute response derivatives. The random global bank is explicitly
a broad-search baseline, not a spectral method. Equal token caps are not equal
total computation; record actual SVD and time separately.

## Fixed panel and stopping

Use eight tasks, Cartesian product of start `[-.5,y,theta]`, `y` in
`{-0.27, 0.29}`, `theta` in `{-0.85, 1.2}`, goal `[.5,gy]`, `gy` in
`{-0.24, 0.2}`. Check exact non-overlap with the v5/v10, waypoint,
response-waypoint, amplitude and remainder development panels; historical
independence beyond these checks is unknown. Use fixed seeds in the runner,
15 steps, `<.05` distance success, and one execution per arm/root. No
outcome-dependent tuning, extensions or retries.

Save all actual actions, outputs, scores, selection and resource ledgers.
Audit the global bank identity between v13 and its matched control, bounds,
token accounting, preserved incumbents and full trajectories against both
exact and independently coded same-environment wall geometry. Report all
failures and stop after the fixed 32 attempts. The experiment cannot itself
pass PC2 or PC3: final-chain interventions and independent strong-baseline
confirmation remain necessary. If this development comparison is negative,
do not infer that reconfiguration or response theory is false.
