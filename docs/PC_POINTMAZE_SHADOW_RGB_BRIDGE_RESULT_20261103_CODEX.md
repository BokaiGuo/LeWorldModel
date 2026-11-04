# RGB-only shadow bridge: development result

The opt-in `ShadowRGBBridge` is a new implementation; it does not replace or
modify the legacy bridge or any historical artifact. A setup authority supplies
the initial state at construction, as in the old bridge. The observation API
returns copied uint8 RGB arrays and a high-level action count; state stays
within the executor/shadow process boundary. Python object privacy is an API
boundary, not a security sandbox.

The executor uses the old pre-command forward and five integration ticks per
action; no extra render-time forward is applied to it. A separate process owns
the shadow sim and GL context. It copies the fields audited in v4, checks copy
equality, forwards only the shadow and returns only encoded RGB and its hash.
The parent validates exact schema, sequence, packet clock, bytes and hash. On
timeout, malformed or extra fields, wrong count, or process failure, it closes
the process group, raises, and refuses reuse; it never retries an uncertain
action. Input validation before dispatch does not poison the bridge.

Host smoke on the first four preselected v4 trajectories completed 136
high-level actions in two passes per trajectory. Across both packet schedules
and repeated reset, all 44 checked RGB hashes and action counts matched the
v4 audit. Receipt:
`le-wm-main/results/pc_pointmaze_shadow_rgb_bridge_smoke_20261103_CODEX.json`;
the companion event journal records every checked boundary.

On the NVIDIA host, 23 tests passed in 5.56s: transport/reset/action-domain and
clock tests; malformed, extra-state, wrong-hash, wrong-sequence, partial and
timed-out replies; fake worker death and process-group cleanup; real shadow
SIGKILL and SIGSTOP followed by fail-closed behavior and cleanup; and the
previous audited state-copy/saved-array regression tests. The first sandbox
attempt could not use local socketpair (`PermissionError`); host tests were
therefore run outside that sandbox. The initial TDD collection failed before
the module existed, as expected.

This is a fixed-model development bridge and smoke test, not a claim that all
64 audited trajectories were run through the new bridge or that planning
improves. It has not been tested on different MuJoCo models, other renderers,
multi-threaded callers beyond rejecting overlap, remote machines, or long
production workloads. The worker checks the XML hash at runtime; code/helper
hashes are recorded and checked by the smoke runner, not enforced by the
bridge itself at runtime. Python private members and IPC are not a security
boundary against a hostile in-process caller.
