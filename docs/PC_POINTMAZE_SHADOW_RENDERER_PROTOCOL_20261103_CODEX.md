# Shadow renderer: pre-run development contract

Freeze the 32 pairs / 64 sealed cp12 action sequences and all 18 observation
boundaries each. This is a post-outcome interface audit, not a planner or an
independent performance evaluation. Reconstruct the actual old executor:
initial forward and the existing pre-command forward remain; **no new forward
is applied to the main simulator**. A separate MjSim with the same frozen XML
is used only for observation; copy the main `MjSimState` (time, qpos, qvel,
act, udd_state), plus ctrl, mocap position/quaternion, applied generalized and
body forces, warmstart and userdata. Require exact copy of every nonempty
field before shadow forward, and exact main fields before/after shadow render.
Record model dimensions and fail closed for nonempty udd_state or model
features whose rendering-relevant state is not covered by this contract.
Only the shadow is forwarded and rendered; move target offscreen after forward.
No hidden state is returned to any planner.

Compare the full main qpos/qvel/time at every one of 86 integration states
against the earlier sealed-image-matched native full-tick replay, not merely
against shadow state. Render the old observation as well and require all 1,152
legacy image hashes match the sealed images. At all 1,152 boundaries compare
all shadow geom positions and orientation matrices with an independently
forwarded reference sim initialized from the same full copied state; additionally
compare particle geom position with its initial qpos offset, tolerance 1e-12.
Record image hashes, every state, geom residuals, copy checks, model dimensions,
GL identity and provenance. A legacy-frame mismatch or any physical-state
mismatch fails; preserve attempted results and errors under new paths.

The independent geometry reference shares MuJoCo and XML and therefore is not
an independent physics backend. A passing replay supports only this exact model,
runtime, host and fixed action panel. A deployable bridge would need a separate
interface and packet-level validation after the gates pass. No performance claim
is permitted. No existing source or result is changed.
