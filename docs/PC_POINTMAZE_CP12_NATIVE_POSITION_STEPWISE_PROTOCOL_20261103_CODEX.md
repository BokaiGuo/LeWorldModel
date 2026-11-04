# Frozen post-outcome native-render versus position-backend audit

Freeze this protocol before new replay. The roster is the 32 branches and
both arms of `pc_pointmaze_cp12_rgb_history_paired_20261103_CODEX`, exactly
64 already sealed 17-command sequences and their original reset states.
Bind the context, decision, execution and evaluation receipts, their NPZs,
the source receipt/NPZ, code, both XML generators, backend executables and
compiled/runtime libraries. Refuse overwrite and changed hashes. No model,
optimization, planning, candidate selection or additional actions beyond
these already sealed sequences.

Run an independent instrumented *replay* of the native bridge's code path:
same official visual XML, `mujoco_py` legacy library, initial reset/forward,
camera and target-site handling, velocity clipping, five integrator ticks per
high-level command, and `sim.render(224,224)` after every command. The
instrumented worker captures qpos[0:2], qvel[0:2] and simulator time after
each render, including initial t0. Compare all 18 RGB hashes for each replay
against the new stream's sealed prefix[0:13] plus executed tail[13:18];
reject a state-equivalence result if any RGB differs. This is a new
instrumented deterministic replay, not access to hidden state from the old
execution process.

Separately replay identical reset/actions in the recorded MuJoCo-2.2
position binding with the physical evaluator's distinct XML and exact
velocity clipping, forward, control clipping and five steps per command.
Capture qpos/qvel and simulator time at t0 through t17. Independently invoke
the existing audited backend's `replay` on every prefix length 0..17, and
require exact equality to the progressive position-backend state at each
time. Require final states to match the previously saved 64 endpoint arrays
bitwise, and each arm's complete actions to match both execution and sealed
decision/context NPZs. Record per-trajectory, per-step position and velocity
component differences and norms, time and final command count; list maxima,
bitwise equality counts and threshold classification changes. Report exact
comparison, not a retrospectively selected tolerance.

Source XMLs and bindings are intentionally different: visual legacy
`mujoco_py` 2.1.2 runtime with rendering vs MuJoCo 2.2 binding with the
position XML without rendering. The same initial numeric state, actions,
high-level command clock and clipping are required, but these checks do not
establish general equivalence, actual historical hidden states, image/state
simultaneity or independent outcomes. State is evaluation-only, in separate
post-decision files; it must never be passed to any planner or used to alter
existing decisions. Record active GL identity, runtime hashes and elapsed
times. Fail closed on any binding, clock, RGB or reference-backend mismatch;
keep partial failure journal. Counts: 64 native RGB replays and 64 progressive
position replays, 1,088 commands each; independent original replay at all
prefixes uses 9,792 commands, excluding empty prefixes.
