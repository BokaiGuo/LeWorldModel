# Frozen RGB geometry clock and forward bridge development protocol

Use the 32 pairs (64 already executed 17-command trajectories) in the
sealed cp12 history-planning execution. Pin its source, decisions, context,
execution and stepwise backend receipts and arrays, the visual XML and
legacy bridge. No planner, model, candidate choice or new actions.

Run two independently reset native-render simulations per trajectory using
the exact reset, 17 commands, per-command velocity clipping, pre-step
forward, control, five `sim.step` ticks and target relocation. The legacy arm
renders without post-step forward. The synchronized arm calls `sim.forward`
after each command, then relocates the target again and renders. At reset and
all 85 tick boundaries record direct qpos, qvel, time and particle geom
world position; at each action boundary record geometry before/after any
extra forward, raw RGB and synchronized RGB. Record the pre-final-tick qpos.
Determine the fixed particle geometry offset from the reset state and verify
the geom mapping from simulator geometry and qpos, not image centroid.
Compare raw boundary geometry with pre-final-tick qpos plus offset, and
forwarded boundary geometry with current qpos plus offset. Report component
residuals, pixel differences and exact/finite checks; do not treat pixels as
ground-truth qpos. Include a stationary or coincident geometry case without
requiring an image difference.

Fail closed unless both simulations have the same qpos, qvel and time at
all 86 boundaries for all trajectories, legacy RGB matches all 1,152 sealed
RGB frames, reset poses and action clocks match, GL is NVIDIA, and source
hashes match. Count exact geometry matches and residuals without claiming
universal equivalence; report no performance implications. Save all numeric
state/geometry/timing arrays, per-frame hashes, pixel counts and predicted
versus measured geometry positions.

If geometry demonstrates cached pre-final-tick positions and the explicit
forward synchronizes the current boundary without changing physical
trajectories, implement an independently named backward-compatible bridge:
default legacy mode delegates to the untouched original, and explicit
`synchronized` mode performs forward and target relocation before rendering.
No dynamic simulator state is exposed to its caller. Validate the new
bridge's legacy and synchronized frames, clock, action packing, reset
behavior and full action trajectories against the isolated instrumented
reference on all 64 sequences; verify legacy matches sealed historical RGB.
Fail on mismatches; preserve old scripts and artifacts. No new evaluation
result may be substituted for the earlier planning outcome. Label
synchronized observations as a new input contract; no claim of improved
planning or checkpoint training alignment.
