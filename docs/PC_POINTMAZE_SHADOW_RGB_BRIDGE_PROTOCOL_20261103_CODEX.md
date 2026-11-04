# Shadow RGB bridge development contract

Pre-implementation contract. This is a new, opt-in PointMaze environment
bridge, not a change to any sealed executor or planner. Its renderer reproduces
the process-isolated shadow mechanism audited in
`pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.json`. Only the fixed
official XML with SHA-256
`64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d`
and the audited legacy runtime are admitted.

An environment setup authority supplies four finite initial state values at
construction, as in the old bridge; these are not a planner observation.
The public observation is only a copied 224×224×3 uint8 RGB array; `step_count`
is the count of high-level actions. `reset()` restores that constructor state,
returns its shadow RGB and resets the count to zero. `step(actions)` accepts
1–5 finite two-component actions in [-1,1]; each action executes the old
velocity clip, old pre-command forward, and five old integration steps, with
no extra main forward. The packet yields the shadow-rendered image after its
last action and must advance the count by exactly its length. Hidden main
state is transferred only inside the worker-to-shadow process boundary and
is never returned on the planner-facing channel.

Use the audited state-copy fields and same model constraints. Validate the
worker's exact response schema, request sequence, action count, image size,
base64 and SHA-256 before returning anything. Invalid local inputs are rejected
before dispatch. Timeout, short or malformed response, wrong clock/sequence,
worker rejection or process death poisons the bridge: stop both worker
processes, raise, return no partial observation, and reject all later calls.
Do not auto-retry uncertain steps. `close()` is idempotent and also kills the
worker process group. Timeouts bound the entire send/receive transaction, not
only receipt of the first byte. Concurrent calls are rejected.

First write fail-closed tests, then implement. Test reset, packet clock,
invalid input, malformed/corrupt replies, wrong clocks, hangs, partial replies,
worker death and child cleanup using isolated test workers. Real-host smoke
uses preselected sealed starts/actions and compares every shadow RGB hash to
the prior v4 audit, including reset and multi-action packet boundaries; it is
not a new planning or performance experiment. Preserve all existing files.
