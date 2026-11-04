# PointMaze obstacle challenge: actual action clock and trace replay

The [fixed replay protocol](PC_POINTMAZE_OBSTACLE_V16_REPLAY_PROTOCOL_20261103.md)
reconstructed all 96 saved controller traces across the 24 exposed,
equal-stratum roots. Every initial and final distance matched its source
report to the fixed `1e-10` final tolerance. At the actual controller
starting state, archive frame 40, all 24 straight segments to the declared
goal intersect the static obstacle under the existing clearance rule; the
same is true at frame 0. This is a geometric diagnostic, not a guarantee
that a particular action must first increase distance.

The old controller's `_step` performs **five MuJoCo integrator steps per
saved block** and then returns a new high-level state. Its 24-block cap is
therefore 24 high-level actions, not 120 high-level actions. The source
report's phrase “5-step pulse” must not be interpreted as five dataset
actions. This replay deliberately uses the source transition, including
its per-block simulator reinitialization; it does not assert equivalence to
the separate persistent `NativeRGBBridge` implementation.

Under the source's original goal-distance `<=0.25` completion criterion,
the privileged one-step, waypoint-conditioned oracle succeeded on 24/24
roots. Its first saved high-level action did not reduce final-goal distance
on 8/24 roots, all eight of which ultimately met that criterion. RGB greedy
also had eight such first actions and succeeded on seven of those eight.
These observations show that the old task contains trajectories with
initial non-improvement and eventual completion; they do not prove that
non-improvement was necessary, changed future control capability, or
generated a Planning Compiler advantage. The oracle uses physical state
and is not a deployable RGB-only planner or perfect long-horizon MPC.

The source roots, goals and trajectories were already exposed. No new
policy was selected, no neural inference occurred, and this is not an
independent confirmation panel. The fixed receipt's 96 trace endpoints,
position arithmetic and source hashes passed one saved-data test.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_obstacle_v16_replay_20261103_DEV01.json)
SHA-256 `d7cea2a8f4b41074f4f48d43a9489f4eb021e9e247d04ffebf4cc88e0dce4c71`.
