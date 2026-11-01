# P5 read-only host and sealed-source preflight

This preflight never initializes CUDA, runs neural inference or launches
physics. It hashes the named instrumented wiring, source chain, checkpoint,
previous RGB bridge and baseline, and uses the existing input preflight to
validate only the two permitted history PNGs per branch. It does not reopen
the physical receipt or future labels. Hence the original receipt's current
bytes are not freshly reverified; the previous saved bindings are checked by
their own pinned artifacts.

For the current sandbox view it reports loaded NVIDIA kernel module version,
installed module version, NVML symlink version, three device-node metadata
checks, project Python/installed torch distribution metadata and available
filesystem space. `modinfo` is metadata-only. Device node and `os.access`
checks do not establish usable GPU access, cgroup permissions, actual CUDA
initialization, available GPU memory, device UUID or MIG state. `/dev/nvidia0`
is a conservative diagnostic, not a selection mechanism for an arbitrary
GPU UUID. The mandatory disk floor is chosen by the reviewer; it is not a
measured P5 storage requirement.

The tool always returns blocked and exit code 2: a later authorized, fresh
device/VRAM/runtime check and an external watchdog plus explicit launch
authorization remain required even if every static check passes. The
instrumented launcher does not yet consume this report as an enforcement
gate; reviewers must inspect both before any future launch. The report is an
exclusive new file, never overwrites frozen results, and no automatic retry
or fallback exists.
