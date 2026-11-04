# Exact wall: secondary controls, exploratory

This protocol was fixed after observing the first kill-test result. It is not
an independent test. It adds two planners to the same start, goal, wall and
12-step action contract:

1. Random bridge: 64 intermediate points uniformly drawn from
   [-0.5,0.5] x [-1.5,1.5], RNG seed 20261103. For each bridge, form the
   polyline start-bridge-goal, distribute 12 steps evenly along polyline arc
   length, project each action to the permitted disk, replay all actions and
   choose the lowest terminal distance. The planner does not inspect wall
   geometry, and all 64 results remain saved.
2. Full-horizon finite-difference L-BFGS-B: four independent random initial
   24-dimensional action sequences with RNG seed 20261104. Each coordinate is
   bounded to [-0.3,0.3], and each 2D action is projected onto the radius-0.3
   disk. Use SciPy L-BFGS-B with at most 30 iterations per start, minimize
   terminal distance, and save every objective invocation, iteration,
   stopping message and selected program. This optimization is not GN/iLQR
   and the wall transition is nonsmooth; a negative result does not show
   differentiable MPC generally fails.

All evaluations use exact transitions and the same 12-step horizon. Track
transition and objective counts for each method; no claim of equal runtime or
total resources. Neither method uses learned representations, and neither
result proves D2P reconfiguration or P1. Do not substitute these
post-pilot controls for an independently fixed benchmark.
