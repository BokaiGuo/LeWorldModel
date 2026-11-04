# Process-isolated PointMaze shadow renderer: development result

Post-outcome replay on the same 32 pairs / 64 already exposed trajectories,
not an independent experiment or performance measurement. The frozen contract
is `PC_POINTMAZE_SHADOW_RENDERER_PROTOCOL_20261103_CODEX.md`; the context and
process addenda were written before their respective reruns. No planner or
model was called, and no existing result or decision was changed.

The v4 main simulator follows the legacy executor, including its initialization
and pre-command forwards, without an additional render-time main forward. A
separate process owns an independent shadow `MjSim` and offscreen GL context.
At each of 1,152 boundaries, a private socket transfers time, qpos, qvel and
all active fields covered by this fixed model, as well as control, applied
forces, warmstart and userdata. Nonempty UDD is rejected. Copy equality,
main state unchanged by shadow observation, and time/qpos/qvel/act preserved
by shadow forward each pass 1,152/1,152 times. The fixed XML has nq=2, nv=2,
na=0, nu=2, nmocap=0, nuserdata=0, neq=0, ntendon=0; this is not a proof of
copy sufficiency for other MuJoCo models or mutable model assets.

Main qpos/qvel/time exactly match the prior sealed-image-matched replay at all
64 × 86 = 5,504 integration states. All 1,152 legacy raw frame hashes match the
sealed historical frames. Shadow geom position and orientation match a second
forwarded simulation from the copied state exactly at every boundary; particle
geometry minus current qpos and the fixed offset has maximum norm
6.280369834735101e-16. The shadow RGB hash matches the prior in-place-forward
diagnostic at 1,152/1,152 boundaries and differs from the old raw RGB at 1,088
post-action boundaries. Image hashes are not used to infer physical position.

The first attempt stopped on a zero-length field exposed as `None`. V2 and V3
executed the full panel but failed the sealed raw image gate (0/1,152); their
negative receipts and arrays remain. V3's explicit same-device context switch
did not repair this. V4's process isolation passed, consistent with a
render-context interference issue, but it does not establish a complete root
cause within the graphics driver. Existing paths and outputs remain untouched.

The second geometry simulator uses the same MuJoCo runtime and XML, not an
independent dynamics backend. The replay does not cover arbitrary starts,
packets, models, devices, action selection or production lifecycle. The v4
worker is an audit implementation with internal state telemetry, **not a
planner-facing or deployable bridge**: a separate RGB-only bridge contract,
packet/reset/timeout/crash tests and binding to this renderer would still be
required. The result supports a candidate observation mechanism, not a claim
about planning performance. Four transfer and saved-result tests passed.
