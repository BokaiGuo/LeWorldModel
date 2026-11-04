# Nonlinear-guarded spectral development experiment

This is a new development method prompted by the weak-direction numerical
audit, not a change to any preregistered gate. For each of the two strongest
and weakest Jacobian modes and each sign, test parameter-space radii 0.5,
0.1, then 0.02. Keep the first radius where the nonlinear frozen-model
displacement has norm at least 1e-5 and differs from the Jacobian prediction
by at most 0.3 relative to the nonlinear displacement. Reject the direction
otherwise. Zero and a separately identified, **unguarded** DLS candidate
remain in the bank. The threshold is a development design choice, not a
scientific success criterion. Float32 is retained; the guard may reject
directions due to precision as well as genuine nonlinearity.

Each retained prefix is evaluated by the same predicted-successor CEM, and
the model-only selector is unchanged. Both selected candidates were guarded
spectral modes, not DLS. Root 20263160 retained four signed strong modes and
no weak modes; root 20263161 retained four strong and two weak signed modes.
The guard consumed 20 and 18 nonlinear predictor rollouts respectively, in
addition to one baseline primal, one Jacobian primal and 192 backward VJP
rows per root. Prefix and suffix selection consumed 396 and 528 predictor
samples respectively. Encoding, replay and wall-time costs are not fully
certified, so this is not an equal-budget performance result.

The frozen model chose both prefixes before physical execution. Each arm
replanned its suffix from its own actual RGB and action history with the
same CEM configuration. The qpos distance below is evaluator-only.

| Seed | Arm | Initial | After prefix | Final |
|---|---|---:|---:|---:|
| 20263160 | selected | 1.003500 | 1.037149 | 0.558267 |
| 20263160 | zero | 1.003500 | 1.010031 | 0.546408 |
| 20263160 | same-bank greedy | 1.003500 | 0.982156 | 0.497367 |
| 20263161 | selected | 1.182600 | 1.267141 | 0.619442 |
| 20263161 | zero | 1.182600 | 1.245444 | 0.598310 |
| 20263161 | same-bank greedy | 1.182600 | 1.148837 | 0.760234 |

Selected worsens the measured immediate qpos distance and is worse than zero
at the final step on both roots. It beats same-bank greedy only on the second
root. There is no common-suffix intervention for these new candidates, no
independent root or full strong-baseline comparison, and no native success
analysis. A model-consistency guard is not evidence that the model matches the
real environment. These results do not establish P1/PC2, nor prove that the
underlying mechanism is impossible.

Artifacts:

- `le-wm-main/results/reacher_pc_guarded_spectral_development_20261102_v1/report.json`
- `le-wm-main/results/reacher_pc_guarded_spectral_physical_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_guarded_spectral.py`

Unit tests cover exact linear acceptance, nonlinear backtracking, and invalid
scale ordering. Preserve the prior unguarded outputs and their negative
physical findings.
