# PointMaze static-geometry persistent-dynamics feasibility development protocol

This is a development feasibility diagnostic, not a Planning Compiler
performance or independent-confirmation experiment. The four starts are the
entire output of the existing static geometry rule: the free points of
`(-0.2,0.8,1.8,2.8,3.8)^2` whose straight segment to goal `(1.8,2.8)`
intersects an obstacle, with clearance 0.08 and 81 segment samples.
Initial velocity is zero. No observed method outcome is used for inclusion.

Use the existing legacy XML and persistent MuJoCo 3.1.6 simulation. A high
level action clips velocity to `[-5,5]`, clips each action coordinate to
`[-1,1]`, then performs five MuJoCo integration steps. Each trajectory
receives at most 24 high-level actions, terminating immediately on position
distance to the goal at most 0.25. Record all positions, velocities, actions,
distance, contacts, and stop states; retain all failures.

Two privileged diagnostic policies operate on true state, using the same
fixed 32 action bank (eight directions times magnitudes .25, .5, .75, 1).
The direct policy greedily minimizes one-step simulated distance to the goal.
The waypoint policy uses shortest four-neighbor BFS over the static free
lattice and greedily minimizes one-step distance to the next waypoint. At
each decision each candidate is scored by a freshly initialized simulator
with current position and velocity; the selected action is executed in the
separate persistent simulator. This is an approximate one-step oracle, not a
perfect planner; its search copies do not preserve simulator warm-start state.
Break ties by bank order. These policies access privileged state and geometry
and must never feed the RGB planner.

Report 3.1.6 as a separate backend; prior 2.2/2.1.2 replay agreement does not
certify this version or these roots. Success establishes only a feasible
trajectory under this particular clock and backend. Wall obstruction,
temporary distance non-improvement, and feasibility do not establish the
necessity or mechanism of non-greedy reconfiguration. No root, threshold,
clock, candidate count, or horizon may be altered based on these outcomes.
