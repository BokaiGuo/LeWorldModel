# CP12 native-render versus position-backend stepwise replay

The prewritten `PC_POINTMAZE_CP12_NATIVE_POSITION_STEPWISE_PROTOCOL_20261103_CODEX.md`
was run post-outcome on all 32 fixed prefixes and both already sealed
single/three-frame tails. Each sequence contains 17 high-level commands.
No planner, model call, selection, or new candidate action was involved.

- **RGB binding:** all 1,152 native replay images (64 × 18, including reset)
  match the saved execution's corresponding new-stream context and tail
  images byte-for-byte.
- **State:** after each image render, all 1,152 native `[qpos_x,qpos_y,
  qvel_x,qvel_y]` vectors match the position-backend vectors exactly at
  float64 precision. Maximum position L2, velocity L2 and component absolute
  difference are all **0.0**.
- **Clock:** each replay finishes at command 17, 85 integration ticks,
  simulator time `0.8500000000000005`; the two recorded time arrays match.
- **Position reference:** the progressive position recorder matched the
  existing audited `replay` function independently restarted at every
  prefix length 0–17. All 64 final states also exactly reproduce the
  previous evaluation NPZ endpoints.
- **Threshold:** there are zero disagreements in the strict `<0.5` endpoint
  threshold between these two replayed state streams.

The sources and actions are numerically identical, but the backends are
**not the same implementation**. Native replay uses the official visual
XML (SHA-256 `64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d`),
`mujoco_py` 2.1.2.14 with the local 2.1.2 runtime, camera and target-site
operations, and NVIDIA OpenGL renders after every command. Position replay
uses a different position XML (SHA-256
`a41452e9b6bd6d55e8e1f746e8e29e4eb1356484ee7161550e51aabbc2ca51c8`),
the MuJoCo 2.2.0 Python binding, and no rendering. Both use velocity clipping
at 5, forward before each command, five ticks per command and dt 0.01.
Inputs are already within [-1,1], making the position backend's explicit
action clipping inactive for these trajectories.

This is an **instrumented deterministic replay** that reproduces every
saved RGB image; the original online bridge did not expose hidden state.
It is not a direct measurement of the hidden state during that old process,
not a proof for untested trajectories or different software, and does not
show that rendered pixels reflect the fully integrated state at the same
instant. The four roots and selected trajectories were already exposed.
Physical states are saved exclusively in the new offline audit, never sent
to a planner or used to change decisions.

Native and progressive position replay each used 1,088 commands. Independent
restarts of the original position function across all 18 prefixes required
9,792 additional commands. No neural inference was performed. The native
worker reports NVIDIA RTX 5060 Ti / driver 595.99.02 directly from its active
OpenGL context. The report binds the 2.1.2 runtime library; a separate
post-run hash check found the native compiled extension SHA-256
`daf43161ab3c8c281e431e8db1b412a126ca74c73498ccdf233b9b360eb3f3dd`
and installed MuJoCo 2.2.0 library SHA-256
`6fe6f954a0bbd5324cf1a310e32be438860e9e602d84269741222179adde67c2`.
Those last two post-run hashes were not bound in the run receipt, so complete
historical library provenance is not certified.

The new report SHA-256 is
`699300a1d2a3566e58d448411d5de857ade7ced506f51fd555147c9d85b75477`.
Focused tests: **10 passed**, including saved-state, clock, RGB, prior action
and endpoint arithmetic checks. They do not themselves rerun either engine.
