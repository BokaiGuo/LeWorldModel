# P5 RGB CUDA supervisor

This is an independent launch wrapper, not permission to run. It checks the
pinned offline review gate, input identities and source hashes, then queries
exactly one physical GPU by UUID with a 10-second query timeout. It rejects MIG,
unknown/unsupported query fields, insufficient total/free MiB, and unsupported
compute mode. Both minimum total and minimum free are explicit reviewer choices.
The free-memory value is a point-in-time admission check, **not a reservation or
an allocation cap**; another process can consume memory immediately afterward.
No model-specific sufficient-memory threshold has been measured.

It fixes `CUDA_VISIBLE_DEVICES` to the admitted UUID and launches exactly once,
without a shell, in a new session/process group; device index 0 is then the
single visible device. An external monotonic deadline covers child startup,
runner preflight, CUDA initialization, and execution. At timeout it sends
SIGKILL to that process group, waits up to five seconds, records partial, and
does not retry. SIGINT/SIGTERM to the supervisor during supervision also request
group termination. If termination is not confirmed, it does not read or seal
the potentially active journal. SIGKILL is an enforcement request, not a
guarantee of prompt completion for uninterruptible kernel waits; a supervisor
itself killed with SIGKILL or host failure cannot enforce or finalize. An
external service/container watchdog is needed for those failures. Normally
exited children are not followed by a process-group kill; the specific runner
does not intentionally spawn descendants, but leaked descendants require
external process supervision.

Each unique run gets an exclusive supervisor directory with fsynced admission,
launch and completion events, stdout and stderr; the existing runner separately
creates its own exclusive journal directory. Timeouts, errors and missing,
truncated, or unverified journals stay partial; existing bytes are not repaired.
Valid journal review requires the strict envelope gate and retains the 164-row
fixed denominator, with separate off/on counts. If preflight fails before the
supervisor directory is created, no durable supervisor receipt is produced.
If the supervisor dies, absence of its final event means incomplete, never
successful. Successful process exit and self-reported journal are not
independent hardware attestation or scientific admission.

After explicit main-thread review and authorization only:

```bash
.venv/bin/python -m experiments.spectral_reachability.p5_rgb_cuda_supervisor_20261030_codex_v1 --authorization MAIN_THREAD_APPROVED_SUPERVISED_FIXED_164_RGB_CUDA --gpu-uuid GPU-<reviewed-uuid> --min-free-mib <reviewed-free> --min-total-mib <reviewed-total> --hard-seconds <reviewed-external-deadline> --runner-seconds <reviewed-between-call-limit> --run-suffix <unique>
```

Tests use mocked GPU query results and ordinary CPU-only subprocesses, not a
GPU or model. No future image, label, simulator, training, or reserved root
is needed by the supervisor.
