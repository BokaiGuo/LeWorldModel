# PointMaze archive fixed-bank replanning oracle

Development-only A0 feasibility diagnostic using the existing sealed four
archive starts (1880–1883), archive-frame-10 positional goal, and the same
eight-action bank as the matched model/physical report. No model inference,
new roots, training, or candidate selection from outcomes is used.

From each saved initial state, compare four privileged simulator policies
over ten archive actions (two packets of five repetitions):

1. `joint_open`: exact initial 8×8 terminal-distance minimizer, then execute
   both chosen packets without replanning.
2. `joint_replan`: same first packet as `joint_open`; after executing it,
   enumerate the same 8×8 two-packet bank from its physical successor and
   execute only its first packet. This has a five-action lookahead beyond the
   common final time and is reported separately, not treated as a same-horizon
   advantage.
3. `greedy_replan`: choose the one-packet positional minimizer and repeat the
   one-packet search after actual execution.
4. `zero`: execute zero actions for ten steps.

All minima use deterministic array-order tie-breaking. Save actual full
backend-returned successor states, all candidate position-cost matrices,
actions, source identity and per-policy backend rollout/step accounting.
Require the initial one/two-packet positions to match the sealed source
within 1e-10 and reject nonfinite values. Early stop/success is not provided
by this endpoint backend: no episode success claim is made.

The backend has a position-only replay contract; its returned velocity and
state-reinitialization equivalence for counterfactual receding-horizon
states are not certified. Thus this is a privileged position-backend
diagnostic, **not** proof of perfect full dynamics, an RGB-only MPC, a strong
fair-compute comparison, independent generalization or P1 admission.
