# Development queue: native goal RGB and observed history

Development-only known-seed replay; confirmation slots cannot execute, and no
independence, training exclusion or P1 admission is asserted. The native
PushT instance supplies `info.goal_state` and `info.goal` from its actual
reset; the latter must equal the same instance's internal goal RGB, is saved
as a PNG and is rechecked on every warmup return. Its file and decoded pixel
hashes bind the reset receipt and the complete history digest carried by the
queue. This is native reset's own goal, not a separately derived target or
proof of target/task suitability.

The same instance executes ten precommitted raw zero actions and records
observed RGB at steps 0, 5 and 10. Each dispatch/return, early stop, failed
attempt, and source receipt retains the existing development queue and
exclusive-reservation semantics. No observation or action is padded; no
neural model is called. The old queue confirmation slot is a non-executable
schema placeholder. History, training membership, complete historical
exclusion, externally trusted timing, access control and provenance remain
unknown. No formal confirmation queue is frozen by this tool.
