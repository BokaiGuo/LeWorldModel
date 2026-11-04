# Reacher spectral candidate and feedback development experiment

This development test replaces the earlier source-packet candidate bank with
directions from the frozen visual model's one-packet action Jacobian. At the
neutral parameter origin it compiles a damped least-squares proposal and both
signs of the two strongest and two weakest right singular directions, with a
fixed 0.5 parameter-space L2 radius. The neutral proposal is retained.
Candidates are evaluated with the same predicted-successor CEM budget per
candidate. The full nonlinear predictor determines the ranking. No simulator
state is an input to candidate selection or suffix planning.

The two source development roots and their RGB goals were rebound by hash.
Model parameters were unchanged. The strongest-direction centered finite
difference relative errors were 0.000139 and 0.000156; this does **not**
validate the weak directions, finite-amplitude approximation, or physical
fidelity. Each 192-by-10 Jacobian was charged as 192 backward VJP rows, plus
one primal rollout and two diagnostic finite-difference rollouts. Selection
reports its own predictive samples separately. Encoder, auxiliary replay and
end-to-end wall costs remain uncertified.

The model-only selector chose a weak-mode direction in each root. Its predicted
suffix-cost advantage over zero was only 0.0000184 and 0.0001177 respectively.
All three physical arms used their own real successor RGB and action history
with the same CEM suffix configuration.

| Seed | Arm | First-packet qpos distance | Final qpos distance |
|---|---|---:|---:|
| 20263160 | selected | 1.010821 | 0.548603 |
| 20263160 | zero | 1.010031 | 0.546408 |
| 20263160 | same-bank greedy | 0.877371 | 0.589113 |
| 20263161 | selected | 1.245720 | 0.599019 |
| 20263161 | zero | 1.245444 | 0.598310 |
| 20263161 | same-bank greedy | 1.081520 | 0.674557 |

Initial distances were 1.003500 and 1.182600. On this evaluator metric the
selected prefix did not improve either initial distance, and its final distance
was worse than zero on both roots, though better than the same-bank greedy.
These are two known development roots, not independent evidence. The same-bank
greedy searches this ten-candidate bank and is not a full-horizon CEM or GRASP
baseline. This experiment has no common-suffix intervention and no native task
success assessment; it neither demonstrates nor disproves real changes in
future controllability. It does not open any gate.

Source artifacts:

- `le-wm-main/results/reacher_pc_spectral_cem_development_20261102_v1/report.json`
- `le-wm-main/results/reacher_pc_spectral_physical_development_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_spectral_candidates.py`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_spectral_cem_development.py`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_spectral_physical_development.py`

The short-term next diagnostic is a predeclared common-suffix intervention
on these exact prefixes, including the predicted and observed response at
matched action scales, before interpreting the small predicted score
differences. A genuine claim still requires independent roots, physical
mechanism evidence, full compute accounting and strong controls.
