# Wall-unicycle random-waypoint secondary control

This new development control is specified after observing that Atlas and
full-horizon CEM failed the single exact wall-unicycle task. It is not
independent confirmation. It uses the same start, goal, command bounds,
16-step horizon and exact analytical transition as the previous protocol.

For each existing seed, draw 207 waypoint coordinates from the fixed box
`[-0.5,0.5] × [-1.5,1.5]`. The range is a manually chosen task-scale
hyperparameter on an exposed problem and cannot be presented as a generic
untuned baseline. Construct a bounded open-loop action program that rotates
in place and traverses a straight line from start to the waypoint and then
to the goal; each segment uses the minimum number of 0.3-or-shorter forward
commands. The construction uses the exact start pose, goal and unicycle
kinematics, but does not inspect the wall or collision feedback when
constructing the program. Wrap angles to `[-pi,pi)`, split turns into minimum
bounded increments, truncate a too-long program at 16 steps or zero-pad a
short program. Evaluate all 207 programs with exact collision dynamics and
select the lowest terminal distance, breaking ties by first index.

The 3,312 search transitions are below the 3,320 observed Atlas transitions
per seed, including its finite-difference Jacobian transitions. This matches
only a transition cap, not other compute, and the control uses more complete
candidate programs and task-space action parameterization. Report candidate
scores and all selected trajectory actions, bounds, contacts, wall time,
and selected-program replay; no waypoint or result is provided to Atlas.
The program builder and replay add work; replay counts are reported
separately. Do not adjust the box or candidate count after outcomes.
