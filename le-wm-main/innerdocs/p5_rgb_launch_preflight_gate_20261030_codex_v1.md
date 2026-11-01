# P5 hash-bound launcher preflight gate

This is an opt-in new entry point, not permission to run. It does not modify the
prior launcher or historical artifacts. It accepts an external SHA-256 for a
canonical, exclusive host preflight report; checks its source pin, fixed input
identities and static blockers, rejects missing device nodes, invalid or stale
UTC timestamps (reviewed maximum 1–300 seconds), version mismatches and
insufficient disk, then repeats the static inspection and compares stable
conditions. The original report remains `ready_for_launch=false`: this wrapper
only gates static prerequisites, not GPU readiness or scientific admission.

It requires the wrapper process to already be in the exact cgroup of a running
systemd service with a bounded `RuntimeMaxUSec`, bounded `TimeoutStopUSec`,
`KillMode=control-group` and `SendSIGKILL=yes`. Unsupported duration formats,
unknown properties or mismatched cgroups fail closed. These are read-only
systemd property checks, not a guarantee the manager or host will survive,
nor a proof against configuration races. SIGKILL can be delayed by
uninterruptible kernel waits.

Only after these checks and an explicit main-thread authorization string does
it create an exclusive fsynced binding receipt containing report SHA, refreshed
conditions, watchdog settings, source hashes and reviewed device/resource
choices. It rechecks report identity, source bytes, freshness and watchdog
settings, then delegates once to the existing instrumented supervisor, which
performs fresh GPU UUID/free-memory/MIG admission and its own timeout.
Preflight is not a memory reservation; races remain. A binding file is not
proof of launch or completion, and may remain if a later check fails. There is
no automatic retry.

The SHA and string are integrity/reviewer intent, **not authentication
credentials**: this code does not cryptographically identify a main thread or
prove who approved the report. An actual execution still needs explicit
authorization and trusted orchestration. The existing launcher remains
directly invocable; this opt-in wrapper cannot enforce policy against callers
that bypass it. The current saved host report has missing device nodes and is
therefore rejected. Do not use the mocked tests as evidence of real GPU,
systemd service behavior or successful production.
