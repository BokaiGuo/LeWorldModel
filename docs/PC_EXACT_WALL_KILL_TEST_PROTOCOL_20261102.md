# Exact wall-detour kill test: development protocol

Fixed before execution. This is a deterministic analytical environment, not a
learned world model or a physical benchmark. A point begins at (-0.5, 0) and
aims at (0.5, 0). Each action lies in the 2D disk of radius 0.3. An action
whose straight-line segment intersects the wall x=0 at |y|<1 is rejected and
leaves the state unchanged; boundary contact is blocked unless |y|>=1.
The terminal cost is Euclidean goal distance after 12 actions.

Any continuous route crossing x=0 through an opening has distance at least
sqrt(1.25)>1 from the goal at crossing, whereas the initial distance is 1.
Thus any successful continuous trajectory must at some point be farther from
the goal than at the start. Rejected transitions do not cross the wall.
This proves a task-level detour necessity, not that a particular local action
is non-greedy or that reconfiguration changes the action Jacobian.

Compare, before examining results:

1. one-step greedy final-goal descent on 16 equally spaced maximal actions
   plus zero, repeated 12 times;
2. full-horizon CEM: 256 samples, 8 rounds, 32 elites, disk-projected Gaussian,
   fixed seed 20261102, retain best full trajectory;
3. deterministic beam search: 16 compass actions plus zero, width 20 for
   12 layers, ranked only by current goal distance with deterministic ties;
4. privileged geometric waypoint witness, which uses the known wall geometry
   and is not a fair learned-model planner.

Report all terminal errors, full trajectories, rejected transitions, and
actual simulated transitions and candidate scores separately. Wall time is
descriptive only; equal candidate counts or transitions do not equate other
operation costs. The witness targets the upper opening with y=1.1, then the
goal, traversing each straight segment using steps of length at most 0.3.
Check every segment, maximum distance and final error. This explicitly
exposes whether the setting is feasible and whether short-horizon goal
scoring stalls even under exact dynamics. It does not test D2P, confirm P1,
or show an advantage over a well-tuned optimizer.
