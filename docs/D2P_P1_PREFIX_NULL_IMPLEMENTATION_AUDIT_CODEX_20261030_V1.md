# Prefix-null exploratory runner and saved audit

The fixed roster contains all eight historical selected plans, each paired with
its exact selected prefix and a five-step all-zero raw prefix. Both roles retain
identical suffix bytes; all 16 logical rows remain in the roster. Duplicate
groups identify only identical task/root/condition/action bytes and do not
authorize execution deduplication. This is known-root, outcome-informed
exploration, not independent confirmation, equal-budget evidence or P1
admission.

The execution prepare revalidates the externally pinned roster and rebuilds
the existing native source closure, including original root, goal, renderer,
warmup source and its file bindings. It binds the new runner, consumer, roster
and inherited adapter files. The current execution prepare is
`le-wm-main/results/codex_p1_prefix_null_runner_20261030_v1_prepare_CODEX03.json`,
SHA-256 `ea27d8e80e1408d9690221195b027c9d4e372c38ac240d430dacec10576ada2d`.
CODEX01 and CODEX02 are superseded because implementation files changed.
The source roster remains deployment-blocked; the runner enforces that block
before environment construction, even if the explicit exploration authorization
string is supplied. A future reviewed deployment decision would require a new
frozen source/implementation and explicit authority; it cannot be obtained
by editing this saved record or passing a flag.

The implementation delegates execution to the existing metered physical adapter:
it uses a distinct bounded output directory, independent environments,
original reset/place/warmup, literal per-step actions, RGB guards, cp5
references and the fixed resource caps. The adapter writes operation intent
before dispatch and keeps attempted/completed counts, failures, native stops
and unstarted row shells. An engineering failure aborts later dispatch; missing
rows or checkpoints are never substituted. If storage or process failure
prevents a final report, the journal or report can itself be incomplete; the
consumer refuses to certify that record.

The separate saved consumer rechecks the sealed prepare and source bindings,
report SHA, journal chain, operation intent/return/failure accounting and all
16 rows. For completed records it additionally runs the previous saved audit,
checks exact requested/actual action bytes, source warmup frames, complete
off/on/off frame triplets, native state/snapshot identity and terminal flags,
cp5 reference bytes, and derives observed checkpoint position/angle/velocity
values without filling missing checkpoints. Failures and partial records remain
`blocked_incomplete`; malformed or unauthenticated artifacts are rejected.
This is saved-record checking, not an independent re-execution or proof of all
hidden simulator state.

CPU-only tests: 33 passed across the new roster and runner/consumer tests and
the inherited runner/episode/sequence tests. They include failed accounting,
out-of-bounds capture, native early stop, missing/tampered inputs, cp5 and frame
coverage. An older physical report was read solely as a saved-format regression
fixture; it is not a result of this intervention. The source checks and tests do
not constitute a deployment certification. No production, physical, or GPU run
was started in this work.
